# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

def plot_image(orig, trans):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(orig)
    axs[0].set_title("Original")
    axs[1].imshow(trans)
    axs[1].set_title("Transformed")
    plt.show()

## ------- Datasets -----
folder = "/datalocal/"
emnist = torchvision.datasets.EMNIST(folder, split="digits", train=False)
idx = 200
img = emnist.data[idx,:,:]
plt.imshow(img)
plt.title(f"Target number: {emnist.targets[idx].item()}")
plt.show()

## ------- Tansforms -----
# trans = transforms.Compose([transforms.CenterCrop(10)])
trans = transforms.Compose([transforms.RandomRotation(45)])
plot_image(img, trans(img.unsqueeze(0))[0,:,:])

## ------- Models ------

