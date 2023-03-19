# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

#%% --- Just for plotting a batch of the mnist dataset
def plot_batch_mnist(batch_imgs, batch_labels):
    batch_size = len(batch_imgs)
    fig, axs = plt.subplots(1, batch_size, figsize=(10,5))
    for i in range(batch_size):
        axs[i].imshow(batch_imgs[i,0,:,:])
        axs[i].set_title(f"Label {batch_labels[i]}")
    plt.show()


#%% EMNIST Dataset
folder = "/data/TORCH_TEST/"
mytransform = transforms.Compose([ transforms.ToTensor() ])
emnist = torchvision.datasets.EMNIST(folder, split="digits", transform=mytransform, download=True)
dataloader = DataLoader(emnist, batch_size=5, shuffle=True)

for c_batch in dataloader:
    x, y = c_batch
    plot_batch_mnist(x, y)
    break
print("Done!")

#%%
def plot_batch(input, output, labels):
    batch_size = len(input)
    fig, axs = plt.subplots(2, batch_size, figsize=(10,5))
    for i in range(batch_size):
        axs[0,i].imshow(input[i,0,:,:], cmap="gray")
        axs[0,i].set_title(f"Input {i}:{labels[i]}")
        axs[1,i].imshow(output[i, 0, :, :], cmap="gray")
        axs[1,i].set_title(f"Pooling {i}:{labels[i]}")
    plt.show()

#%%
class CNNModel(nn.Module):
    # On the init function we define our model
    def __init__(self):
        super().__init__()  # Constructor of parent class
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)


    # On the foreward model we indicate how to make one 'pass' of the model
    def forward(self, x):
        return self.pool2(self.pool1(x))

#%%
model = CNNModel()
x, y = next(iter(dataloader))
output = model(x)
plot_batch(x, output, y)
