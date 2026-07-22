import os
import re
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for headless/cluster runs
import matplotlib.pyplot as plt

import pennylane as qml


# =====================================================================================
# NOTE: The model architecture, fixed scene, and key hyperparameters below must stay in
# sync with the training script. They're duplicated here so this file can run
# standalone without depending on the training script's module path.
# =====================================================================================

n_inputs = 2
n_outputs = 1
n_neurons = 35

# QML Params (must match train_one_scene_model.py)
n_qubits = n_inputs
n_qlayers = 3

# Data-generation hyperparameters (used for ground-truth comparison fields)
m, n = 12, 12
beta = 1.67
Abase = 0.7
phase_seed = 42  # must match phase_seed in train_one_scene_model.py

# Fixed scene parameters -- must match train_one_scene_model.py exactly, since the
# model was only ever trained on (x, y) points drawn from this one scene.
scene_L = 27.5
scene_C_threshold = 0.475
scene_phase = np.random.default_rng(7).uniform(0, 2 * np.pi)


# =====================================================================================
# QML model: a hybrid quantum-classical network (see train_one_scene_model.py for the
# full explanation). Duplicated here, architecture-for-architecture, so checkpoints
# saved by the training script load cleanly into this standalone module.
# =====================================================================================

_qdevice = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(_qdevice, interface="torch", diff_method="backprop")
def _quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class FourierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(_quantum_circuit, weight_shapes)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, n_outputs),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        q_out = self.qlayer(x)
        return self.head(q_out)


# =====================================================================================
# Checkpoint utilities
# =====================================================================================

def find_latest_checkpoint(directory='.', pattern=r'model_1_(\d+)\.pt'):
    """
    Scan a directory for checkpoints named like 'model_1_{epoch}.pt' (as saved by the
    training script whenever validation loss improves) and return the path to the one
    with the highest epoch number. Returns None if no matching files are found.
    """
    candidates = []
    regex = re.compile(pattern)
    for path in glob.glob(os.path.join(directory, '*.pt')):
        match = regex.search(os.path.basename(path))
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_model(model_path, device='cpu'):
    """
    Instantiate a FourierModel, load the given state_dict checkpoint into it,
    move it to the requested device, and set it to eval mode.
    """
    model = FourierModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# =====================================================================================
# Model-driven field generation
# =====================================================================================

def generate_grid_inputs(L, resolution=200):
    """
    Build a (resolution x resolution) grid of normalized (x, y) coordinates in [0, 1],
    ready to feed straight into the model -- since the scene is fixed, (x, y) are the
    only two features it needs.

    Returns:
        inputs_tensor: (resolution*resolution, 2) float32 tensor, ready for the model
        X_norm, Y_norm: (resolution, resolution) meshgrid arrays (normalized coords),
                         kept around so outputs can be reshaped/plotted later
    """
    lin = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    X_norm, Y_norm = np.meshgrid(lin, lin)

    flat_x = X_norm.ravel()
    flat_y = Y_norm.ravel()

    inputs = np.stack([flat_x, flat_y], axis=1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    return inputs_tensor, X_norm, Y_norm


def generate_cloud_field(model, L, resolution=200, device='cpu', batch_size=8192):
    """
    Use a trained model to predict cloud density over a full (resolution x resolution)
    grid for the fixed scene size L.

    Returns a dict with:
        'X_norm', 'Y_norm' : (resolution, resolution) normalized coordinate grids
        'X', 'Y'            : (resolution, resolution) physical coordinate grids (x*L, y*L)
        'cloud_density'     : (resolution, resolution) predicted density field in [0, 1]
        'L'                 : the scene's L
    """
    inputs_tensor, X_norm, Y_norm = generate_grid_inputs(L, resolution)

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, inputs_tensor.shape[0], batch_size):
            chunk = inputs_tensor[start:start + batch_size].to(device)
            out = model(chunk)
            preds.append(out.cpu())

    preds = torch.cat(preds, dim=0).numpy().reshape(resolution, resolution)

    return {
        'X_norm': X_norm,
        'Y_norm': Y_norm,
        'X': X_norm * L,
        'Y': Y_norm * L,
        'cloud_density': preds,
        'L': L,
    }


# =====================================================================================
# Ground-truth field generation (same Fourier synthesis as training, but evaluated on
# a regular grid instead of random samples, for direct visual/quantitative comparison)
# =====================================================================================

def generate_ground_truth_field(L, C_threshold, scene_phase, resolution=200, m=m, n=n, beta=beta,
                                 Abase=Abase, phase_seed=phase_seed):
    """
    Recompute the analytic Fourier cloud field (same generative process used to build
    the training data) over a (resolution x resolution) grid for the fixed scene's L,
    C_threshold, and scene_phase. Useful as a ground-truth counterpart to
    generate_cloud_field's output.

    Returns a dict with the same shape/keys as generate_cloud_field, so the two can be
    compared or plotted side by side.
    """
    lin = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    X_norm, Y_norm = np.meshgrid(lin, lin)
    x_coords = (X_norm * L).ravel()
    y_coords = (Y_norm * L).ravel()

    C_xy = np.full(x_coords.shape[0], Abase, dtype=np.float64)

    # Same fixed per-component base phases (seeded) plus the scene's fixed phase
    # offset used by train_one_scene_model.py's generate_data.
    base_phi = np.random.default_rng(phase_seed).uniform(0, 2 * np.pi, size=(2 * m + 1, 2 * n + 1))

    for mi in range(-m, m + 1):
        for ni in range(-n, n + 1):
            if mi == 0 and ni == 0:
                continue

            k = np.sqrt(mi**2 + ni**2)
            A_mn = k**(-beta / 2)
            phi = base_phi[mi + m, ni + n] + scene_phase

            wave_angle = (2 * np.pi * mi * x_coords) / L + (2 * np.pi * ni * y_coords) / L + phi
            C_xy += A_mn * np.cos(wave_angle)

    c_min, c_max = C_xy.min(), C_xy.max()
    if c_max != c_min:
        relative_humidity = (C_xy - c_min) / (c_max - c_min)
    else:
        relative_humidity = np.zeros_like(C_xy)

    cloud_density = np.clip(relative_humidity - C_threshold, 0, None)
    if cloud_density.max() > 0:
        cloud_density = cloud_density / cloud_density.max()

    cloud_density = cloud_density.reshape(resolution, resolution)

    return {
        'X_norm': X_norm,
        'Y_norm': Y_norm,
        'X': X_norm * L,
        'Y': Y_norm * L,
        'cloud_density': cloud_density,
        'L': L,
        'C_threshold': C_threshold,
    }


# =====================================================================================
# Quantitative comparison utilities
# =====================================================================================

def compute_field_metrics(predicted_field, true_field):
    """
    Compare a predicted cloud density field against a ground-truth field of the same
    shape. Returns a dict of scalar metrics useful for reporting or later annotating
    plots (e.g. as a title or legend entry).
    """
    pred = np.asarray(predicted_field)
    true = np.asarray(true_field)

    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs true {true.shape}")

    diff = pred - true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))

    pred_flat = pred.ravel()
    true_flat = true.ravel()
    if np.std(pred_flat) > 0 and np.std(true_flat) > 0:
        correlation = float(np.corrcoef(pred_flat, true_flat)[0, 1])
    else:
        correlation = float('nan')

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'correlation': correlation,
    }


def compute_dataset_metrics(model, data_loader, loss_function=None, device='cpu'):
    """
    Run the model over an entire DataLoader (e.g. the test set from the training
    script) and return aggregate metrics: average loss (if a loss_function is given),
    MAE, and MSE across all samples.
    """
    if loss_function is None:
        loss_function = torch.nn.BCELoss()

    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            batch_size_actual = inputs.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_abs_error += torch.sum(torch.abs(outputs - labels)).item()
            total_sq_error += torch.sum((outputs - labels) ** 2).item()
            total_samples += batch_size_actual

    return {
        'avg_loss': total_loss / total_samples,
        'mae': total_abs_error / total_samples,
        'mse': total_sq_error / total_samples,
        'num_samples': total_samples,
    }


# =====================================================================================
# Example usage: plot predicted vs. ground-truth for the one fixed scene
# =====================================================================================


def main():
    checkpoint_dir = './models/qml_one_scene/'
    checkpoint_path = find_latest_checkpoint(directory=checkpoint_dir)

    if checkpoint_path is None:
        print("No checkpoint found matching 'model_1_*.pt' in", checkpoint_dir)
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path)

    # There is only one scene -- generate its predicted and ground-truth fields once.
    predicted = generate_cloud_field(model, scene_L, resolution=150)
    ground_truth = generate_ground_truth_field(scene_L, scene_C_threshold, scene_phase, resolution=150)

    metrics = compute_field_metrics(predicted['cloud_density'], ground_truth['cloud_density'])
    print(f"Scene: L={scene_L:.2f}, C_threshold={scene_C_threshold:.3f}, "
          f"MSE={metrics['mse']:.5f}, MAE={metrics['mae']:.5f}, "
          f"corr={metrics['correlation']:.3f}")

    # Plot predicted vs. ground-truth vs. difference for this scene
    predicted_field = predicted['cloud_density']
    true_field = ground_truth['cloud_density']
    diff_field = predicted_field - true_field

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(predicted_field, origin='lower', cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Predicted')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(true_field, origin='lower', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    max_abs_diff = np.max(np.abs(diff_field)) if np.max(np.abs(diff_field)) > 0 else 1.0
    im2 = axes[2].imshow(diff_field, origin='lower', cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title('Difference (Pred - True)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"L={scene_L:.0f}, C_threshold={scene_C_threshold:.3f} | "
                 f"MSE={metrics['mse']:.5f}, MAE={metrics['mae']:.5f}, corr={metrics['correlation']:.3f}")
    fig.tight_layout()

    save_path = 'scene_one_scene.png'
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
