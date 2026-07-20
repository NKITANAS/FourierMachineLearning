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
    samples = 10000
    scenes  = 1000
    test_scenes = 200
    test_samples = 1000
    batch_size = 1000
 
    m, n             = 12, 12
    beta             = 1.67
    phase_seed       = 42         # Seeds the fixed per-wave-component base phases (see below)
    L_min, L_max     = 5, 50
    Cth_min, Cth_max = 0.30, 0.65
    Abase            = 0.7        # Ranges from (0; 1). Higher = Wetter atmosphere.
 
    # Training Params
    epochs = 2000
    lr     = 0.01
 
    # Model Params
    n_inputs = 5
    n_outputs = 1
    n_neurons = 35
else:
    # TODO: Take arguments from the command line and assign those to the model rather than the values above
    pass
 
 
# Generate Data #
 
def generate_data(samples, scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_seed, Abase):
    # Rename bounds for clarity in the math loop: m and n are the maximum limits (M, N)
    M_limit = m
    N_limit = n

    # Give every (mi, ni) wave component its own random phase, exactly like the full
    # train_model.py does. A single shared `phi` for every component (the earlier
    # "constant phi for testing" version) collapses the sum into a highly symmetric
    # pattern that's identical (in normalized x/L, y/L coordinates) for every scene,
    # so the model has almost nothing spatial to learn and converges to predicting
    # roughly the mean field everywhere -- i.e. a solid rectangle.
    # These base phases are seeded once (not redrawn per scene) so scenes stay a
    # reproducible, learnable function of (x, y, L, C) -- see the per-scene phase
    # offset below for how scenes still get genuinely different realizations.
    base_phi = np.random.default_rng(phase_seed).uniform(
        0, 2 * np.pi, size=(2 * M_limit + 1, 2 * N_limit + 1))

    # Precompute the (mi, ni) wave-component grid once -- it's identical for every
    # scene. Flattening it lets the double loop over components below collapse into
    # a single vectorized (component x sample) broadcast per scene instead of up to
    # (2*M_limit+1)*(2*N_limit+1) separate Python iterations, each doing its own
    # array op over `samples` points. That nested-loop version was the dominant cost
    # of data generation (thousands of Python-level iterations per scene).
    mi_range = np.arange(-M_limit, M_limit + 1)
    ni_range = np.arange(-N_limit, N_limit + 1)
    MI, NI = np.meshgrid(mi_range, ni_range, indexing='ij')
    component_mask = ~((MI == 0) & (NI == 0))  # drop the DC component (covered by Abase)
    mi_flat = MI[component_mask].astype(np.float64)
    ni_flat = NI[component_mask].astype(np.float64)
    k_flat = np.sqrt(mi_flat**2 + ni_flat**2)
    A_flat = k_flat ** (-beta / 2)
    base_phi_flat = base_phi[component_mask]

    X_chunks, Y_chunks = [], []
    L_chunks, L_norm_chunks = [], []       # REAL L / normalized L
    C_norm_chunks, Phase_norm_chunks = [], []
    C_xy_chunks = []

    for scene in range(scenes):
        # 1. Define random L and C_threshold for this specific scene
        L = np.random.uniform(L_min, L_max)
        C_threshold = np.random.uniform(Cth_min, Cth_max)
        L_norm = (L - L_min) / (L_max - L_min)
        C_norm = (C_threshold - Cth_min) / (Cth_max - Cth_min)
        # NOTE: L and C_threshold are intentionally kept as their own (real) variables
        # here rather than being overwritten by L_norm/C_norm — everything below that
        # does physical math (sampling coordinates, computing wave_angle, thresholding)
        # must use the real values. Only L_norm/C_norm get passed to the model as inputs.

        # Random per-scene phase offset, added on top of every component's fixed base
        # phase below. This is what actually makes each scene a distinct realization
        # instead of a rescaled copy of the same fixed pattern. It's fed to the model
        # as its own input (Phase_norm) so the mapping stays learnable despite varying
        # from scene to scene.
        scene_phase = np.random.uniform(0, 2 * np.pi)
        scene_phase_norm = scene_phase / (2 * np.pi)

        # 2. Generate random spatial coordinate points within the boundaries of L
        x_coords = np.random.uniform(0, L, size=samples)
        y_coords = np.random.uniform(0, L, size=samples)

        # 3-4. Evaluate the double summation over the grid of wavenumbers for every
        # sample at once. Equation per component: A_mn * cos((2*pi*mi*x)/L + (2*pi*ni*y)/L + phi)
        # wave_angle has shape (n_components, samples); summing over axis 0 performs
        # the same accumulation the nested Python loop used to do, but in one
        # vectorized (BLAS-backed) pass instead of thousands of tiny ones.
        phi = base_phi_flat + scene_phase
        wave_angle = (2 * np.pi / L) * (
            mi_flat[:, None] * x_coords[None, :] + ni_flat[:, None] * y_coords[None, :]
        ) + phi[:, None]
        C_xy = Abase + (A_flat[:, None] * np.cos(wave_angle)).sum(axis=0)

        # 5. Normalize this scene's raw field to a clean [0, 1] range
        c_min, c_max = C_xy.min(), C_xy.max()
        if c_max != c_min:
            relative_humidity = (C_xy - c_min) / (c_max - c_min)
        else:
            relative_humidity = np.zeros_like(C_xy)

        # 6. Apply condensation threshold rules (uses the REAL C_threshold, not C_norm)
        # Density is 0 if it doesn't cross C_threshold. If it does, we scale the remainder.
        cloud_density = np.clip(relative_humidity - C_threshold, 0, None)
        if cloud_density.max() > 0:
            cloud_density = cloud_density / cloud_density.max()

        # 7. Collect this scene's arrays; concatenated once at the end instead of
        # growing Python lists element-by-element (avoids boxing every float and the
        # final list->ndarray conversion cost over millions of rows).
        X_chunks.append(x_coords)
        Y_chunks.append(y_coords)
        L_chunks.append(np.full(samples, L))
        L_norm_chunks.append(np.full(samples, L_norm))
        C_norm_chunks.append(np.full(samples, C_norm))
        Phase_norm_chunks.append(np.full(samples, scene_phase_norm))
        C_xy_chunks.append(cloud_density)

    return (
        np.concatenate(X_chunks), np.concatenate(Y_chunks),
        np.concatenate(L_chunks), np.concatenate(L_norm_chunks),
        np.concatenate(C_norm_chunks), np.concatenate(Phase_norm_chunks),
        np.concatenate(C_xy_chunks),
    )
 
 
# Create Dataset #
 
class FourierDataset(dataset.Dataset):
    def __init__(self, X, Y, L, L_norm, C_norm, Phase_norm, C_xy):
        # Convert lists to numpy arrays
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        L = np.array(L, dtype=np.float32)             # REAL L, used only to normalize X/Y below
        L_norm = np.array(L_norm, dtype=np.float32)    # normalized L, goes into the model input
        C_norm = np.array(C_norm, dtype=np.float32)    # normalized C_threshold, goes into the model input
        Phase_norm = np.array(Phase_norm, dtype=np.float32)  # normalized scene phase, goes into the model input

        # 1. Coordinate Normalization (Crucial!)
        # Scale X and Y relative to their specific scene's REAL L.
        # Now (X_norm, Y_norm) are always strictly bounded between 0 and 1.
        X_norm = X / L
        Y_norm = Y / L

        # 2. Stack inputs: [X_norm, Y_norm, L_norm, C_norm, Phase_norm]
        # All five features are now consistently scaled to comparable ranges,
        # which is what the model actually needs to see.
        inputs = np.stack([X_norm, Y_norm, L_norm, C_norm, Phase_norm], axis=1)
 
        # Convert to tensors directly in RAM
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.out = torch.tensor(C_xy, dtype=torch.float32).view(-1, 1)
 
    def __len__(self):
        return len(self.inputs)
 
    def __getitem__(self, idx):
        return self.inputs[idx], self.out[idx]
        # ADD THIS NEW METHOD HERE 
    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.L = self.L.to(device)
        self.L_norm = self.L_norm.to(device)
        self.C_norm = self.C_norm.to(device)
        self.Phase_norm = self.Phase_norm.to(device)
        self.C_xy = self.C_xy.to(device)
        return self # Allows chaining like dataset.to(device) 
 
 
 
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
    running_loss = 0.0
    last_loss = 0.0

    # Shuffle once per epoch via a GPU-resident permutation, then slice batches
    # directly out of the tensors already sitting in VRAM. This replaces the
    # DataLoader (which paid Python-level __getitem__/collate overhead per sample,
    # plus a host->device copy per batch) with plain tensor indexing -- the same
    # shuffled-minibatch SGD, just without the per-batch interpreter overhead.
    perm = torch.randperm(inputs.shape[0], device=inputs.device)

    for i, start in enumerate(range(0, inputs.shape[0], batch_size)):
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

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000.0  # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.0

    return last_loss
 
 
def main():
    # Create the data
    # 1. Setup device target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 2. Create the data normally (Leave them as standard lists/arrays)
    Xtrain, Ytrain, Ltrain, Ltrain_norm, Ctrain_norm, Phasetrain_norm, C_xytrain = generate_data(
        samples, scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_seed, Abase)
    Xtest, Ytest, Ltest, Ltest_norm, Ctest_norm, Phasetest_norm, C_xytest = generate_data(
        test_samples, test_scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_seed, Abase)

    # 3. Build datasets on CPU...
    training_data = FourierDataset(Xtrain, Ytrain, Ltrain, Ltrain_norm, Ctrain_norm, Phasetrain_norm, C_xytrain)
    test_data = FourierDataset(Xtest, Ytest, Ltest, Ltest_norm, Ctest_norm, Phasetest_norm, C_xytest)

    # 4. ...then move the whole dataset onto the GPU once, up front. Everything fits
    # comfortably in VRAM, so there's no need to shuttle individual batches over from
    # host memory on every step via a DataLoader -- that was the previous (unused)
    # intent of pin_memory=True, except the tensors were never actually placed on
    # `device` before, so training silently ran on CPU regardless of GPU availability.
    train_inputs = training_data.inputs.to(device)
    train_targets = training_data.out.to(device)
    test_inputs = test_data.inputs.to(device)
    test_targets = test_data.out.to(device)

    # Build model, loss, optimizer
    model = FourierModel().to(device)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    os.makedirs('models/cml', exist_ok=True)

    epoch_number = 0
    best_vloss = float('inf')

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
            model_path = f'models/cml/model_1_{epoch_number}.pt'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
 
    # Always save the model's state once training completes, regardless of whether
    # the final epoch happened to be the best one. Without this, if the last epoch
    # isn't an improvement over best_vloss, its weights are never written to disk.
    final_model_path = 'model_1_final_cml.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Training complete. Final model saved to {final_model_path}')
 
 
if __name__ == '__main__':
    main()

