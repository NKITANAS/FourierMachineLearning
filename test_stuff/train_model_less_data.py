# Imports #
import numpy as np
import matplotlib.pyplot as plt
 
import torch
from torch.utils.data import dataset, dataloader
 
# Set Hyperparameters #
 
# Toggle to get params from notebook
params_from_cli = False
 
if not params_from_cli:
    # Data Generation and Processing
    samples = 1000
    scenes  = 1000
    test_scenes = 20
    test_samples = 10000
    batch_size = 100
 
    m, n             = 12, 12
    beta             = 1.67
    phase_seed       = 42         # Seeds the fixed per-wave-component base phases (see below)
    L_min, L_max     = 5, 50
    Cth_min, Cth_max = 0.30, 0.65
    Abase            = 0.7        # Ranges from (0; 1). Higher = Wetter atmosphere.
 
    # Training Params
    epochs = 2000
    lr     = 0.001
 
    # Model Params
    n_inputs = 5
    n_outputs = 1
    n_neurons = 35
else:
    # TODO: Take arguments from the command line and assign those to the model rather than the values above
    pass
 
 
# Generate Data #
 
def generate_data(samples, scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_seed, Abase):
    # Initialize lists to store the final dataset rows
    X_out = []
    Y_out = []
    L_out = []       # REAL (physical) L, needed later to correctly normalize X/Y
    L_norm_out = []   # normalized L in [0, 1], this is what the model actually sees
    C_norm_out = []   # normalized C_threshold in [0, 1], also a model input feature
    Phase_norm_out = []  # normalized per-scene phase offset in [0, 1], also a model input feature
    C_xyOUT = []

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

        # 3. Initialize the cloud density field for all samples with the baseline offset (Abase / a0)
        # We use a numpy array of size (samples,) to compute everything in parallel
        C_xy = np.full(samples, Abase, dtype=float)

        # 4. Evaluate the double summation over the grid of wavenumbers:
        # m runs from -M_limit to +M_limit, n runs from -N_limit to +N_limit
        for mi in range(-M_limit, M_limit + 1):
            for ni in range(-N_limit, N_limit + 1):
                # Skip the DC component (0,0) as it is already covered by Abase
                if mi == 0 and ni == 0:
                    continue

                # Calculate frequency wave magnitude k
                k = np.sqrt(mi**2 + ni**2)

                # Calculate amplitude component A_mn
                A_mn = k**(-beta / 2)

                # Each wave component keeps its own (seeded, reproducible) base phase,
                # shifted by this scene's random phase offset.
                phi = base_phi[mi + M_limit, ni + N_limit] + scene_phase

                # Compute the spatial wave contribution for all coordinate points simultaneously
                # Equation: A_mn * cos((2*pi*mi*x)/L + (2*pi*ni*y)/L + phi)
                # Note: uses the REAL L (domain size) to correctly scale the spatial period.
                wave_angle = (2 * np.pi * mi * x_coords) / L + (2 * np.pi * ni * y_coords) / L + phi
                C_xy += A_mn * np.cos(wave_angle)
 
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
 
        # 7. Append the calculated coordinates and outputs to our global lists
        X_out.extend(x_coords.tolist())
        Y_out.extend(y_coords.tolist())
        L_out.extend([L] * samples)              # real L, for coordinate normalization later
        L_norm_out.extend([L_norm] * samples)      # normalized L, for the model's input feature
        C_norm_out.extend([C_norm] * samples)      # normalized C_threshold, for the model's input feature
        Phase_norm_out.extend([scene_phase_norm] * samples)  # normalized phase, for the model's input feature
        C_xyOUT.extend(cloud_density.tolist())

    # Return the flat lists ready to be loaded into your dataset object
    return X_out, Y_out, L_out, L_norm_out, C_norm_out, Phase_norm_out, C_xyOUT
 
 
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
 
 
def train_one_epoch(training_loader, model, optimizer, loss_function):
    running_loss = 0.0
    last_loss = 0.0
 
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
 
        # Zero your gradients for every batch!
        optimizer.zero_grad()
 
        # Make predictions for this batch
        outputs = model(inputs)
 
        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
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
    Xtrain, Ytrain, Ltrain, Ltrain_norm, Ctrain_norm, Phasetrain_norm, C_xytrain = generate_data(
        samples, scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_seed, Abase)
    Xtest, Ytest, Ltest, Ltest_norm, Ctest_norm, Phasetest_norm, C_xytest = generate_data(
        test_samples, test_scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_seed, Abase)

    training_data = FourierDataset(Xtrain, Ytrain, Ltrain, Ltrain_norm, Ctrain_norm, Phasetrain_norm, C_xytrain)
    test_data = FourierDataset(Xtest, Ytest, Ltest, Ltest_norm, Ctest_norm, Phasetest_norm, C_xytest)
 
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
 
    # Build model, loss, optimizer
    model = FourierModel()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
 
    epoch_number = 0
    best_vloss = float('inf')
 
    for epoch in range(epochs):
        print(f'EPOCH {epoch_number + 1}:')
 
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(training_loader, model, optimizer, loss_function)
 
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
 
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(testing_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss.item()
 
        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
 
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_1_{epoch_number}.pt'
            torch.save(model.state_dict(), model_path)
 
        epoch_number += 1
 
    # Always save the model's state once training completes, regardless of whether
    # the final epoch happened to be the best one. Without this, if the last epoch
    # isn't an improvement over best_vloss, its weights are never written to disk.
    final_model_path = 'model_1_final.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Training complete. Final model saved to {final_model_path}')
 
 
if __name__ == '__main__':
    main()

