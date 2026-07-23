# Torch for optimization
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# the quantum circuit
def quantum_circuit(datapoint, params):
    # Encode the input data as an RX rotation
    qml.RX(datapoint, wires=0)
    # Create a rotation based on the angles in "params"
    qml.Rot(params[0], params[1], params[2], wires=0)
    # We return the expected value of a measurement along the Z axis
    return qml.expval(qml.PauliZ(wires=0))

# Classical Pre/Postprocessing
def loss_func(predictions, Y):
    # This is a postprocessing step. Here we use a least squares metric
    # based on the predictions of the quantum circuit and the outputs
    # of the training data points.

    total_losses = 0
    for i in range(len(Y)):
        output = Y[i]
        prediction = predictions[i]
        loss = (prediction - output)**2
        total_losses += loss
    return total_losses

# Define your cost function, including any classical pre/postprocessing
def model_qc(params, qnode, X):
    # We get the predictions of the quantum circuit for a specific
    # set of parameters along the entire input dataset
    predictions = torch.stack([qnode(x, params)  for x in X])
    return predictions

# ## Class: Dataset definition
class Dataset_1D(Dataset):
    # load the dataset
    def __init__(self, datafull_IP, datafull_OP):   
        # store all the inputs and outputs
        self.X = datafull_IP
        self.Y = datafull_OP

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

def prepare_data(datafull_IP, datafull_OP, n_val, batch_size, pin_memory=False):
    # load the dataset
    dataset = Dataset_1D(datafull_IP, datafull_OP)
    # calculate split
    train, val = dataset.get_splits(n_val)
    # prepare data loaders
    # ============cuda settings: pinned memory============
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True)
    val_dl   = DataLoader(val  , batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return train_dl, val_dl

class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, n_qubits = None, device = None):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        # Encoding layer(s): pre-PQC layer(s)
        self.pre_net = nn.Linear(n_inputs, n_hidden)
        self.pre_net2 = nn.Linear(n_hidden, n_hidden)
        self.pre_net3 = nn.Linear(n_hidden, n_hidden)
        self.act = nn.LeakyReLU(0.1)
        #self.act2 = nn.Tanh()
        self.act2 = nn.SiLU()
        
        # Decoding layer(s): post-PQC layer(s)
        self.post_net = nn.Linear(n_hidden, n_outputs)

        self.device=device

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = input_features
        pre_out = self.pre_net(input_features)
        pre_out = self.act(pre_out)

        pre_out = self.pre_net2(pre_out)
        pre_out = self.act(pre_out)

        pre_out = self.pre_net3(pre_out)
        pre_out = self.act2(pre_out)

        # return the two-dimensional prediction from the postprocessing layer
        post_out = self.post_net(pre_out)
        #post_out = self.act(post_out)
        return post_out

def model_SingleMLP(n_inputs, n_hidden, n_outputs, n_qubits, device_name):
    model = MLP(n_inputs, n_hidden, n_outputs, n_qubits, device_name).to(device_name)
    print(model)
    print("Single layer MLP with number of parameters = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

class QC(nn.Module):
    """
    https://github.com/olcf/hands-on-with-odo/tree/master/challenges/Python_QML_Basics
    """
    def __init__(self, n_inputs, n_outputs, n_qubits, q_params, qnode, device):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()        
        # Parameters for the PQC
        self.q_params = nn.Parameter(q_params)
        self.qnode = qnode

        self.device=device

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = input_features

        # Apply the quantum circuit to each element of the batch and append to q_out
        # q_out = torch.Tensor(0, pre_out.shape[-1])
        # q_out = q_out.to(self.device)
        # for elem in pre_out:
        #     # q_out_elem = torch.stack(self.qnode(elem, self.q_params)).float().unsqueeze(0)
        #     q_out_elem = self.qnode(elem, self.q_params).unsqueeze(0)
        #     q_out = torch.cat((q_out, q_out_elem))
        q_out = torch.stack(
                            [self.qnode(elem, self.q_params) for elem in pre_out]
                            )

        # return the two-dimensional prediction from the postprocessing layer
        post_out = q_out
        return post_out
    
def model_SingleQC(n_inputs, n_outputs, n_qubits, q_params, qnode, device_name):
    model = QC(n_inputs, n_outputs, n_qubits, q_params, qnode, device_name).to(device_name)
    print(model)
    print("Single PQC with number of parameters = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

class QMLP(nn.Module):
    """
    https://github.com/olcf/hands-on-with-odo/tree/master/challenges/Python_QML_Basics
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, n_qubits, q_params, qnode, device = None):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        # Encoding layer(s): pre-PQC layer(s)
        self.pre_net1 = nn.Linear(n_inputs, n_hidden)
        self.pre_net2 = nn.Linear(n_hidden, n_hidden)
        self.pre_net3 = nn.Linear(n_hidden, n_hidden)
        #self.pre_net3 = nn.Linear(n_qubits, n_hidden)
        #self.pre_net4 = nn.Linear(n_hidden, n_qubits)
        self.act1 = nn.LeakyReLU(0.1)
        self.act2 = nn.Tanh()

        # Parameters for the PQC
        self.q_params = nn.Parameter(q_params)
        self.qnode = qnode
        
        # Decoding layer(s): post-PQC layer(s)
        self.post_net = nn.Linear(n_hidden, n_outputs)

        self.device=device
        self.printed_shapes = False

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        pre_out = self.pre_net1(input_features)
        pre_out = self.act1(pre_out)
        pre_out = self.pre_net2(pre_out)
        pre_out = self.act2(pre_out)
        #pre_out = self.act1(pre_out)
        pre_out = self.pre_net3(pre_out)
        #pre_out = self.act1(pre_out)
        #pre_out = self.pre_net4(pre_out) 
        #pre_out = self.act2(pre_out)
        

        # pre_out shape: (batch_size, n_hidden)
        batch_size, hidden_size = pre_out.shape

        # Flatten all batch and hidden values into one broadcasting dimension
        quantum_inputs = pre_out.reshape(-1)

        # One QNode call instead of one call per sample
        q_flat = self.qnode(quantum_inputs, self.q_params)

        # Restore the original batch structure
        q_out = q_flat.reshape(batch_size, hidden_size)

        post_out = self.post_net(q_out)
        return post_out
    
        # --- Quantum Circuit (PQC) ---
        # Apply the quantum circuit to each element of the batch
        #q_out = torch.stack(
            #[self.qnode(elem, self.q_params) for elem in pre_out]
        #)

        post_out = self.post_net(q_out)
        #post_out = self.act1(post_out)
        
        #return post_out
    
def model_SingleQMLP(n_inputs, n_hidden, n_outputs, n_qubits, q_params, qnode, device_name):
    model = QMLP(n_inputs = n_inputs, n_hidden = n_hidden, n_outputs = n_outputs, n_qubits = n_qubits, q_params = q_params, qnode = qnode, device = device_name).to(device_name)
    print(model)
    print("Single layer QMLP with number of parameters = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def defNNmodel(model_name, n_inputs, n_hidden, n_outputs, n_qubits, q_params, qnode, device_name='cpu'):
    if   model_name == 'SingleMLP':
        model = model_SingleMLP(n_inputs, n_hidden, n_outputs, n_qubits, device_name)
    elif model_name == 'SingleQC':
        model = model_SingleQC(n_inputs, n_outputs, n_qubits, q_params, qnode, device_name)
    elif model_name == 'SingleQMLP':
        model = model_SingleQMLP(n_inputs, n_hidden, n_outputs, n_qubits, q_params, qnode, device_name)
    else:
        print("Enter valid model name...")
    return model
