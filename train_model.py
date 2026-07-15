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
    samples = 100000
    scenes  = 300
    test_scenes = 30
    test_samples = 100000
    batch_size = 4

    m, n             = 12, 12
    beta             = 1.67
    phase_angle      = 1.27       # Can be between 0 and 2*pi
    L_min, L_max     = 5, 50
    Cth_min, Cth_max = 0.30, 0.65
    Abase            = 0.7        # Ranges from (0; 1). Higher = Wetter atmosphere.

    # Training Params
    epochs = 20
    lr     = 0.01

    # Model Params
    n_inputs = 4
    n_outputs = 1
    n_neurons = 100 
else:
    # TODO: Take arguments from the command line and assign those to the model rather than the values above
    pass


# Generate Data #

def generate_data(samples, scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_angle, Abase):
    # Initialize lists to store the final dataset rows
    X_out = []
    Y_out = []
    L_out = []
    C_out = []
    C_xyOUT = []

    # Rename bounds for clarity in the math loop: m and n are the maximum limits (M, N)
    M_limit = m
    N_limit = n

    for scene in range(scenes):
        # 1. Define random L and C_threshold for this specific scene
        L = np.random.uniform(L_min, L_max)
        C_threshold = np.random.uniform(Cth_min, Cth_max)

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

                # Generate a random phase angle shift unique to this specific wave component
                # (Alternatively, you can use the static 'phase_angle' argument if preferred)
                phi = np.random.uniform(0, 2 * np.pi)

                # Compute the spatial wave contribution for all coordinate points simultaneously
                # Equation: A_mn * cos((2*pi*mi*x)/L + (2*pi*ni*y)/L + phi)
                # Note: We divide by L (domain size) to correctly scale the spatial period.
                wave_angle = (2 * np.pi * mi * x_coords) / L + (2 * np.pi * ni * y_coords) / L + phi
                C_xy += A_mn * np.cos(wave_angle)

        # 5. Normalize this scene's raw field to a clean [0, 1] range
        c_min, c_max = C_xy.min(), C_xy.max()
        if c_max != c_min:
            relative_humidity = (C_xy - c_min) / (c_max - c_min)
        else:
            relative_humidity = np.zeros_like(C_xy)

        # 6. Apply condensation threshold rules
        # Density is 0 if it doesn't cross C_threshold. If it does, we scale the remainder.
        cloud_density = np.clip(relative_humidity - C_threshold, 0, None)
        if cloud_density.max() > 0:
            cloud_density = cloud_density / cloud_density.max()

        # 7. Append the calculated coordinates and outputs to our global lists
        X_out.extend(x_coords.tolist())
        Y_out.extend(y_coords.tolist())
        L_out.extend([L] * samples)
        C_out.extend([C_threshold] * samples)
        C_xyOUT.extend(cloud_density.tolist())

    # Return the flat lists ready to be loaded into your dataset object
    return X_out, Y_out, L_out, C_out, C_xyOUT


# Create Dataset #

class FourierDataset(dataset.Dataset):
    def __init__(self, X, Y, L, C, C_xy):
        # Convert lists to numpy arrays
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        L = np.array(L, dtype=np.float32)
        C = np.array(C, dtype=np.float32)
        
        # 1. Coordinate Normalization (Crucial!)
        # Scale X and Y relative to their specific scene's L.
        # Now (X_norm, Y_norm) are always strictly bounded between 0 and 1.
        X_norm = X / L
        Y_norm = Y / L
        
        # 2. Stack inputs: [X_norm, Y_norm, L, C]
        # L and Cth remain in their raw scales so the model knows the physical size 
        # and baseline threshold of the current scene.
        inputs = np.stack([X_norm, Y_norm, L, C], axis=1)

        # Convert to tensors directly in RAM
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.out = torch.tensor(C_xy, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.out[idx]


# Create the data
Xtrain, Ytrain, Ltrain, Ctrain, C_xytrain = generate_data(     samples,      scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_angle, Abase)
Xtest,  Ytest,  Ltest,  Ctest,  C_xytest  = generate_data(test_samples, test_scenes, m, n, beta, L_min, L_max, Cth_min, Cth_max, phase_angle, Abase)

training_data = FourierDataset(Xtrain, Ytrain, Ltrain, Ctrain, C_xytrain)
test_data     = FourierDataset(Xtest,  Ytest,  Ltest,  Ctest,  C_xytest)

training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_loader  = torch.utils.data.DataLoader(test_data,     batch_size=batch_size, shuffle=True)

# Train the Model #

# Define Model
class FourierModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Create Neural Network
        # NOTE: SeLU can be replaced with GeLU.
        self.network = torch.nn.Sequential(

            torch.nn.Linear(n_inputs, n_neurons),  # Input
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_neurons), # Hidden
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_neurons), # Hidden
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_neurons), # Hidden
            torch.nn.SELU(),

            torch.nn.Linear(n_neurons, n_outputs), # Output
            torch.nn.Sigmoid() # make the output range only from 0 to 1

        )

    def forward(self, x):
        return self.network(x)

model = FourierModel()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def train_one_epoch(epoch_index):
    running_loss = 0
    last_loss = 0

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
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            running_loss = 0.

    return last_loss


epoch_number = 0  
best_vloss = 1_000_000.

for epoch in range(epochs):
    print(f'EPOCH {epoch_number + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)


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
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')

    # Log the running loss averaged per batch
    # for both training and validation

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'model_1_{epoch_number}'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
