# Imports #
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import dataset, dataloader

import pennylane as qml

# Set Hyperparameters #

# Toggle to get params from notebook
params_from_cli = False

if not params_from_cli:
    # Data Generation and Processing
    samples = 20000          # training points sampled from the one fixed scene
    test_samples = 5000       # held-out points sampled from that same scene
    batch_size = 500

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
    epochs = 500
    lr     = 0.01

    # Model Params
    n_inputs = 2
    n_outputs = 1
    n_neurons = 35

    # QML Params
    n_qubits = n_inputs   # one qubit per input feature, angle-embedded
    # Depth of the variational circuit AND number of data re-uploads (see
    # _quantum_circuit below). A single AngleEmbedding can only represent frequency
    # +-1 per qubit (Schuld et al. 2021, "quantum models as Fourier series"), which
    # structurally cannot express this field's harmonics up to wavenumber 12 no
    # matter how long training runs -- re-uploading the inputs n_qlayers times raises
    # the reachable Fourier degree to roughly n_qlayers. 8 layers only reached
    # corr~0.66 against ground truth; 16 got to corr~0.77 with no sign of
    # overfitting (validation loss kept dropping alongside train loss), so we
    # spend the extra circuit depth here rather than leaving reachable frequencies
    # off the table.
    n_qlayers = 16
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

class FourierDataset(dataset.Dataset):
    def __init__(self, X, Y, L, C_xy):
        # Convert lists to numpy arrays
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        # Coordinate Normalization (Crucial!)
        # Scale X and Y relative to the scene's fixed L.
        # Now (X_norm, Y_norm) are always strictly bounded between 0 and 1.
        X_norm = X / L
        Y_norm = Y / L

        # Stack inputs: [X_norm, Y_norm] -- the only two features the model sees,
        # since L / C_threshold / phase are constant for this one scene.
        inputs = np.stack([X_norm, Y_norm], axis=1)

        # Convert to tensors directly in RAM
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.out = torch.tensor(C_xy, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.out[idx]




# Train the Model


# =====================================================================================
# QML model: a hybrid quantum-classical network.
#
# Both (already-normalized, roughly-[0,1]) input features are angle-embedded onto
# one qubit each, entangled through a stack of StronglyEntanglingLayers (the quantum
# analogue of the classical hidden layers), then read out as Pauli-Z expectation values
# (one per qubit, each in [-1, 1]). A small classical Linear+Sigmoid head — mirroring the
# output stage of the classical FourierModel — maps those expectation values down to a
# single cloud-density prediction in [0, 1].
# =====================================================================================

_qdevice = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(_qdevice, interface="torch", diff_method="backprop")
def _quantum_circuit(inputs, weights):
    # Data re-uploading: re-embed the 2 input features before every entangling
    # block instead of just once. Inputs are already ~[0, 1] (normalized x, y
    # coordinates), so scale to [0, pi] to use the full range of the RY rotation.
    # Repeating the embedding n_qlayers times is what lets the circuit represent
    # frequencies beyond +-1 (see the n_qlayers comment above).
    for l in range(n_qlayers):
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(n_qubits))

    # Read out one expectation value per qubit.
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class FourierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}

        # Quantum variational layer, wrapped so it behaves like any other
        # torch.nn.Module and can sit inside a Sequential/optimizer as usual.
        self.qlayer = qml.qnn.TorchLayer(_quantum_circuit, weight_shapes)

        # Classical read-out head: maps the n_qubits expectation values
        # (each in [-1, 1]) down to a single cloud-density prediction in [0, 1].
        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, n_outputs),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        q_out = self.qlayer(x)
        return self.head(q_out)


def train_one_epoch(inputs, targets, batch_size, model, optimizer, loss_function):
    # Accumulate over every batch in the epoch and average at the end, instead of
    # only snapshotting the running loss every 1000 batches. With a single fixed
    # scene there are far fewer than 1000 batches per epoch (e.g. 20000 samples /
    # 500 batch_size = 40 batches), so that snapshot condition never fired and
    # last_loss silently stayed at its initial 0.0 for the entire run -- the
    # training loss WAS being computed and backpropagated correctly (see
    # LOSS valid tracking working fine), only the printed number was wrong.
    total_loss = 0.0
    n_batches = 0

    # Shuffle once per epoch and slice batches directly out of the in-memory
    # tensors. This replaces the DataLoader (which paid Python-level
    # __getitem__/collate overhead on every row per epoch) with plain tensor
    # indexing -- same shuffled-minibatch SGD, far less interpreter overhead
    # surrounding each (expensive) quantum circuit call.
    perm = torch.randperm(inputs.shape[0])

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
    # Create the data. Both train and test draws come from the exact same fixed
    # scene (scene_L, scene_C_threshold, scene_phase) -- only the sampled (x, y)
    # points differ.
    Xtrain, Ytrain, C_xytrain = generate_data(
        samples, scene_L, scene_C_threshold, scene_phase, m, n, beta, phase_seed, Abase)
    Xtest, Ytest, C_xytest = generate_data(
        test_samples, scene_L, scene_C_threshold, scene_phase, m, n, beta, phase_seed, Abase)

    training_data = FourierDataset(Xtrain, Ytrain, scene_L, C_xytrain)
    test_data = FourierDataset(Xtest, Ytest, scene_L, C_xytest)

    # Batch by indexing the in-memory tensors directly (see train_one_epoch) instead
    # of going through a DataLoader. NOTE: no device transfer here -- default.qubit's
    # circuit simulation has no AMD/ROCm-accelerated backend (PennyLane's GPU
    # simulators, e.g. lightning.gpu, are NVIDIA/cuQuantum-only), so the quantum layer
    # runs on CPU regardless; moving just the tiny classical head to the GPU would add
    # host<->device round-trips for no benefit.
    train_inputs, train_targets = training_data.inputs, training_data.out
    test_inputs, test_targets = test_data.inputs, test_data.out

    # Build model, loss, optimizer. Adam (vs. plain SGD+momentum) converges to a
    # comparable-or-better minimum in roughly a third of the epochs on this
    # variational circuit -- its per-parameter adaptive step sizes cope much
    # better with the circuit's flat/rugged loss landscape than a single global
    # learning rate does.
    model = FourierModel()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs('models/qml_one_scene', exist_ok=True)

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
            model_path = f'models/qml_one_scene/model_1_{epoch_number}.pt'
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    # Always save the model's state once training completes, regardless of whether
    # the final epoch happened to be the best one. Without this, if the last epoch
    # isn't an improvement over best_vloss, its weights are never written to disk.
    final_model_path = 'model_1_final_qml_one_scene.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Training complete. Final model saved to {final_model_path}')


if __name__ == '__main__':
    main()
