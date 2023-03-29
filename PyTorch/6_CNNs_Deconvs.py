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

def plot_batch(input, output, title, kernel):
    batch_size = len(input)
    fig_size = 8
    fig, axs = plt.subplots(2, batch_size+1, figsize=(fig_size*batch_size, fig_size))

    for i in range(batch_size):
        img1 = axs[0,i].imshow(input[i,0,:,:], cmap='summer')
        img1.set_clim(0, 8)
        axs[0, i].set_xticks(range(input.shape[3]))
        axs[0, i].set_yticks(range(input.shape[2]))
        plt.colorbar(img1, ax=axs[0, i], shrink=0.7)
        # Loop over data dimensions and create text annotations.
        for j in range(input.shape[2]):
            for k in range(input.shape[3]):
                text = axs[0, i].text(k, j, input[i, 0, j, k],
                               ha="center", va="center", color="k")

        img2 = axs[1,i].imshow(output[i, 0, :, :], cmap='summer')
        img2.set_clim(0, 8)
        axs[1, i].set_xticks(range(output.shape[3]))
        axs[1, i].set_yticks(range(output.shape[2]))
        plt.colorbar(img2, ax=axs[1,i], shrink=0.7)
        for j in range(output.shape[2]):
            for k in range(output.shape[3]):
                text = axs[1, i].text(k, j, output[i, 0, j, k],
                               ha="center", va="center", color="k")

    ki = axs[0,i+1].imshow(kernel[0,0,:,:])
    axs[0, i+1].set_xticks(range(kernel.shape[3]))
    plt.colorbar(ki, ax=axs[0, i+1], shrink=0.7)
    axs[0,i+1].set_title("kernel")
    ki = axs[1,i+1].imshow(kernel[0,0,:,:])
    axs[1, i+1].set_xticks(range(kernel.shape[3]))
    plt.colorbar(ki, ax=axs[1, i+1], shrink=0.7)
    axs[1,i+1].set_title("kernel")

    plt.suptitle(title)
    plt.show()

#%%
ksize = 3
stride = 2
inpad = 0
outpad = 0
dilation = 1
operation = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                               kernel_size=ksize,
                               dilation=dilation,
                               stride=stride, padding=inpad, output_padding=outpad, bias=False)
# operation.weight = nn.Parameter(torch.zeros_like(operation.weight))
operation.weight = nn.Parameter(torch.ones_like(operation.weight))
# operation.weight = nn.Parameter(torch.randn_like(operation.weight))

x = torch.randn(2, 1, 2, 2)
x = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]], [[[1.0, 2.0], [1.0, 2.0]]]])
output = operation(x)
print(output)
title = f"ksize={ksize}, stride={stride}, inpad={inpad}, outpad={outpad} \n in{tuple(x.shape)}, out{tuple(output.shape)}"
plot_batch(x.detach().numpy(), output.detach().numpy(), title, kernel=operation.weight.detach().numpy())
