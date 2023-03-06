# # Hello PyTorch Neural Networks
# This notebook show a second level **hello world** example of neural networks. In this case we use NN to approximate any **smooth** function.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time as t

device = "cuda" if torch.cuda.is_available() else "cpu"

## Linear regression Example
# 1. Generate some synthetic data

# Create some synthetic data
x = torch.linspace(-np.pi,np.pi,100)
m = 2
# y = x**2 + torch.rand(x.shape)*2  # Quadratic
y = x**2 + np.sin(x) - 2*np.cos(3*x) + torch.rand(x.shape)*2  # Harmonic

plt.scatter(x,y, s=10)
plt.title("Training noisy data")
plt.show()

## We are now testing three models
# 1. Single neuron model
# 2. Mulitple neurons in a single hidden layer with ReLu for the non-linearity
# 3. Multiple Hidden layers with ReLu and Batch Normalization

# Models are created by classes that inherit from Module
class SingleNeuronModel(nn.Module):
    # On the init function we define our model
    def __init__(self):
        super().__init__() # Constructor of parent class
        self.ex_model= nn.Sequential(
            nn.Linear(1, 1)
        )
    
    # On the forward function we indicate how to make one 'pass' of the model
    def forward(self, x):
        return self.ex_model(x)

class MultipleNeuronModelSingleHiddenLayer(nn.Module):
    # On the init function we define our model
    def __init__(self):
        super().__init__() # Constructor of parent class

        self.neurons_per_layer = 20
        self.input_layer = nn.Linear(1, self.neurons_per_layer)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.neurons_per_layer, 1)
    
    # On the forward function we indicate how to make one 'pass' of the model
    def forward(self, x):
        # l1 = self.relu(self.input_layer(x))  # With simple non-linear function
        l1 = self.relu(self.input_layer(x))  # With simple non-linear function
        l2 = self.output_layer(l1)
        return l2
    
class MultipleNeuronModelMultipleHiddenLayer(nn.Module):
    # On the init function we define our model
    def __init__(self):
        super().__init__() # Constructor of parent class
        self.n_hidden_layers = 2 # 2, 20
        self.neurons_per_layer = 20  # 10, 50
        self.input_layer = nn.Linear(1, self.neurons_per_layer)
        self.bn = nn.BatchNorm1d(self.neurons_per_layer)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons_per_layer, self.neurons_per_layer) for x in range(self.n_hidden_layers)])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.neurons_per_layer, 1)
    
    # On the forward function we indicate how to make one 'pass' of the model
    def forward(self, x):
        l1 = self.input_layer(x)  # With simple non-linear function
        for i in range(self.n_hidden_layers):
            l1 = self.bn(self.relu(self.hidden_layers[i](l1)))  # With batch normalization
            # l1 = self.relu(self.hidden_layers[i](l1))  # With batch normalization
        l2 = self.output_layer(l1)
        return l2

## Choose your model
# ex_model = SingleNeuronModel().to(device)
# ex_model = MultipleNeuronModelSingleHiddenLayer().to(device)
ex_model = MultipleNeuronModelMultipleHiddenLayer().to(device)
print("Total number of parameters: ", sum(p.numel() for p in ex_model.parameters() if p.requires_grad))

# print(list(ex_model.named_parameters()))
# Reshape to the proper input for PyTorch
X = torch.reshape(x, (x.shape[0],1)).to(device)
Y = torch.reshape(y, (y.shape[0],1)).to(device)

#-------------- Just for plotting --------------
fig, ax = plt.subplots(1,1)
def plotCurrentModel(x, y, model, ax):
    # Torch receives inputs with shape [Examples, input_size]
    model_y = model(X).cpu().detach().numpy()

    ax.scatter(x, y, s=10, label='True')
    ax.plot(x, model_y, label='Model', c='r')
    ax.set_title('Default model')
    ax.legend()
    
plotCurrentModel(x, y, ex_model, ax)
plt.show()

## ---- Do Training
loss_mse = nn.MSELoss() # Define loss function
optimizer = torch.optim.SGD(ex_model.parameters(), lr=2e-3) # Define optimization algorithm

# Optimize the parameters several times
ex_model.train()
for i in range(500):
    optimizer.zero_grad()
    pred = ex_model(X)
    loss = loss_mse(pred, Y)
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # ---------- Just for plotting ---------
    if i % 40 == 0:
        fig, ax = plt.subplots(1,1)
        title = f"Iteration number {i} loss: {loss:0.3f}"
        print(title)
        plotCurrentModel(x, y, ex_model, ax)
        plt.show()
        t.sleep(.1)

plotCurrentModel(x, y, ex_model, ax)
plt.show()
print("Done!")


##


##

