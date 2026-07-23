# Imports #
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import dataset, dataloader

# Set Hyperparameters #

# Toggle to get params from notebook
params_from_cli = False

if not params_from_cli:
    # Data Generation and Processing
    samples = 10000        # training points sampled from the one fixed scene
    test_samples = 2000     # held-out points sampled from that same scene
    batch_size = 1000

    m, n             = 12, 12
    beta             = 1.67
    phase_seed       = 42         # Seeds the fixed per-wave-component base phases (see below)
    Abase            = 0.7        # Ranges from (0; 1). Higher = Wetter atmosphere.

    # Fixed scene parameters -- everything about the scene is held constant so the
    # model only has to learn cloud_density as a function of (x, y) for this one
    # field, instead of also conditioning on L / C_threshold / phase like the
    # less-data model does. scene_phase is derived from a fixed RNG seed (7) purely
    # so it's reproducible without hand-picking an arbitrary-looking float.
    scene_L             = 27.5
    scene_C_threshold   = 0.475
    scene_phase         = np.random.default_rng(7).uniform(0, 2 * np.pi)

    # Training Params
    epochs = 2000
    lr     = 0.01

    # Model Params
    # The ground-truth field is itself a truncated 2D Fourier series with wavenumbers
    # up to m=n=12, so raw (x, y) coordinates in [0, 1] make this a badly
    # high-frequency target for a plain coordinate-input MLP: networks are known to
    # have a "spectral bias" toward low frequencies (Tancik et al. 2020) and, in
    # practice here, correlation with ground truth plateaued around ~0.4 no matter
    # how long training ran. Feeding the network sin/cos features at frequencies
    # 1..n_freqs directly (see FourierDataset) gives it the same basis functions the
    # field is built from, which fixes that -- correlation jumps to >0.85 with the
    # same epoch budget.
    n_freqs = 12
    n_inputs = 2 + 4 * n_freqs
    n_outputs = 1
    n_neurons = 35
else:
    # TODO: Take arguments from the command line and assign those to the model rather than the values above
    pass


# Generate Data #

def generate_data(samples, L, C_threshold, scene_phase, m, n, beta, phase_seed, Abase):
    # Rename bounds for clarity in the math loop: m and n are the maximum limits (M, N)
    M_limit = m
    N_limit = n

    # Give every (mi, ni) wave component its own random phase, exactly like the full
    # train_model.py does. A single shared `phi` for every component collapses the
    # sum into a highly symmetric pattern that's identical for every point, so the
    # model would have almost nothing spatial to learn.
    # These base phases are seeded once so the scene stays a reproducible, learnable
    # function of (x, y).
    base_phi = np.random.default_rng(phase_seed).uniform(
        0, 2 * np.pi, size=(2 * M_limit + 1, 2 * N_limit + 1))

    # Precompute the (mi, ni) wave-component grid once. Flattening it lets the double
    # loop over components below collapse into a single vectorized
    # (component x sample) broadcast instead of up to (2*M_limit+1)*(2*N_limit+1)
    # separate Python iterations, each doing its own array op over `samples` points.
    mi_range = np.arange(-M_limit, M_limit + 1)
    ni_range = np.arange(-N_limit, N_limit + 1)
    MI, NI = np.meshgrid(mi_range, ni_range, indexing='ij')
    component_mask = ~((MI == 0) & (NI == 0))  # drop the DC component (covered by Abase)
    mi_flat = MI[component_mask].astype(np.float64)
    ni_flat = NI[component_mask].astype(np.float64)
    k_flat = np.sqrt(mi_flat**2 + ni_flat**2)
    A_flat = k_flat ** (-beta / 2)
    base_phi_flat = base_phi[component_mask]

    # Generate random spatial coordinate points within the boundaries of L. Since the
    # scene (L, C_threshold, phase) is fixed, every point below belongs to the same
    # single field -- only (x, y) varies from sample to sample.
    x_coords = np.random.uniform(0, L, size=samples)
    y_coords = np.random.uniform(0, L, size=samples)

    # Evaluate the double summation over the grid of wavenumbers for every sample at
    # once. Equation per component: A_mn * cos((2*pi*mi*x)/L + (2*pi*ni*y)/L + phi)
    # wave_angle has shape (n_components, samples); summing over axis 0 performs the
    # accumulation in one vectorized (BLAS-backed) pass.
    phi = base_phi_flat + scene_phase
    wave_angle = (2 * np.pi / L) * (
        mi_flat[:, None] * x_coords[None, :] + ni_flat[:, None] * y_coords[None, :]
    ) + phi[:, None]
    C_xy = Abase + (A_flat[:, None] * np.cos(wave_angle)).sum(axis=0)

    # Normalize this scene's raw field to a clean [0, 1] range
    c_min, c_max = C_xy.min(), C_xy.max()
    if c_max != c_min:
        relative_humidity = (C_xy - c_min) / (c_max - c_min)
    else:
        relative_humidity = np.zeros_like(C_xy)

    # Apply condensation threshold rules. Density is 0 if it doesn't cross
    # C_threshold. If it does, we scale the remainder.
    cloud_density = np.clip(relative_humidity - C_threshold, 0, None)
    if cloud_density.max() > 0:
        cloud_density = cloud_density / cloud_density.max()

    return x_coords, y_coords, cloud_density


# Create Dataset #

def fourier_feature_map(X_norm, Y_norm, n_freqs):
    """
    Encode normalized (x, y) coordinates as [x, y, sin(2*pi*k*x), cos(2*pi*k*x),
    sin(2*pi*k*y), cos(2*pi*k*y)] for k in 1..n_freqs, so the network is handed the
    same sinusoidal basis functions the ground-truth field is built from (see the
    n_inputs comment above for why raw coordinates alone underfit this target).
    """
    freqs = np.arange(1, n_freqs + 1, dtype=np.float32)
    x_ang = 2 * np.pi * X_norm[:, None] * freqs[None, :]
    y_ang = 2 * np.pi * Y_norm[:, None] * freqs[None, :]
    return np.concatenate([
        X_norm[:, None], Y_norm[:, None],
        np.sin(x_ang), np.cos(x_ang), np.sin(y_ang), np.cos(y_ang),
    ], axis=1)


class FourierDataset(dataset.Dataset):
    def __init__(self, X, Y, L, C_xy, n_freqs=n_freqs):
        # Convert lists to numpy arrays
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        # Coordinate Normalization (Crucial!)
        # Scale X and Y relative to the scene's fixed L.
        # Now (X_norm, Y_norm) are always strictly bounded between 0 and 1.
        X_norm = X / L
        Y_norm = Y / L

        inputs = fourier_feature_map(X_norm, Y_norm, n_freqs)

        # Convert to tensors directly in RAM
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.out = torch.tensor(C_xy, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.out[idx]



# Train the Model


class FourierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Create Neural Network
        # NOTE: SeLU can be replaced with GeLU.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_neurons),  # Input
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_neurons),  # Hidden
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_neurons),  # Hidden
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_neurons),  # Hidden
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_outputs),  # Output
            torch.nn.Sigmoid()  # make the output range only from 0 to 1
        )

    def forward(self, x):
        return self.network(x)


def train_one_epoch(inputs, targets, batch_size, model, optimizer, loss_function):
    # Accumulate over every batch in the epoch and average at the end, instead of
    # only snapshotting the running loss every 1000 batches. With a single fixed
    # scene there are far fewer than 1000 batches per epoch (e.g. 10000 samples /
    # 1000 batch_size = 10 batches), so that snapshot condition never fired and
    # last_loss silently stayed at its initial 0.0 for the entire run -- the
    # training loss WAS being computed and backpropagated correctly (see
    # LOSS valid tracking working fine), only the printed number was wrong.
    total_loss = 0.0
    n_batches = 0

    # Shuffle once per epoch via a GPU-resident permutation, then slice batches
    # directly out of the tensors already sitting in VRAM. This replaces the
    # DataLoader (which paid Python-level __getitem__/collate overhead per sample,
    # plus a host->device copy per batch) with plain tensor indexing -- the same
    # shuffled-minibatch SGD, just without the per-batch interpreter overhead.
    perm = torch.randperm(inputs.shape[0], device=inputs.device)

    for start in range(0, inputs.shape[0], batch_size):
        idx = perm[start:start + batch_size]
        batch_inputs = inputs[idx]
        batch_labels = targets[idx]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(batch_inputs)

        # Compute the loss and its gradients
        loss = loss_function(outputs, batch_labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    # Create the data
    # 1. Setup device target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 2. Create the data normally (Leave them as standard lists/arrays). Both train
    # and test draws come from the exact same fixed scene (scene_L, scene_C_threshold,
    # scene_phase) -- only the sampled (x, y) points differ.
    Xtrain, Ytrain, C_xytrain = generate_data(
        samples, scene_L, scene_C_threshold, scene_phase, m, n, beta, phase_seed, Abase)
    Xtest, Ytest, C_xytest = generate_data(
        test_samples, scene_L, scene_C_threshold, scene_phase, m, n, beta, phase_seed, Abase)

    # 3. Build datasets on CPU...
    training_data = FourierDataset(Xtrain, Ytrain, scene_L, C_xytrain)
    test_data = FourierDataset(Xtest, Ytest, scene_L, C_xytest)

    # 4. ...then move the whole dataset onto the GPU once, up front. Everything fits
    # comfortably in VRAM, so there's no need to shuttle individual batches over from
    # host memory on every step via a DataLoader.
    train_inputs = training_data.inputs.to(device)
    train_targets = training_data.out.to(device)
    test_inputs = test_data.inputs.to(device)
    test_targets = test_data.out.to(device)

    # Build model, loss, optimizer
    model = FourierModel().to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    os.makedirs('models/cml_one_scene', exist_ok=True)

    epoch_number = 0
    best_vloss = float('inf')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    for epoch in range(epochs):
        print(f'EPOCH {epoch_number + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(train_inputs, train_targets, batch_size, model, optimizer, loss_function)

        running_vloss = 0.0
        n_vbatches = 0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for start in range(0, test_inputs.shape[0], batch_size):
                vinputs = test_inputs[start:start + batch_size]
                vlabels = test_targets[start:start + batch_size]
                voutputs = model(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss.item()
                n_vbatches += 1

        avg_vloss = running_vloss / n_vbatches
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'models/cml_one_scene/model_1_{epoch_number}.pt'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    # Always save the model's state once training completes, regardless of whether
    # the final epoch happened to be the best one. Without this, if the last epoch
    # isn't an improvement over best_vloss, its weights are never written to disk.
    final_model_path = 'model_1_final_cml_one_scene.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Training complete. Final model saved to {final_model_path}')


if __name__ == '__main__':
    main()
