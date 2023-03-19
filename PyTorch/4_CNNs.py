# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

## --- Just for plotting a batch of the mnist dataset
def plot_batch_mnist(batch_imgs, batch_labels):
    batch_size = len(batch_imgs)
    fig, axs = plt.subplots(1, batch_size, figsize=(10,5))
    for i in range(batch_size):
        axs[i].imshow(batch_imgs[i,0,:,:])
        axs[i].set_title(f"Label {batch_labels[i]}")
    plt.show()


## EMNIST Dataset
folder = "/datasetMNIST/"
mytransform = transforms.Compose([ transforms.ToTensor() ])
emnist = torchvision.datasets.EMNIST(folder, split="digits", transform=mytransform, download=True)

##
emnist = torchvision.datasets.CIFAR10(folder,  download=True)

# ## ----- DataLoader MNIST--------
dataloader = DataLoader(emnist, batch_size=5, shuffle=True)

for c_batch in dataloader:
    x, y = c_batch
    plot_batch_mnist(x, y)
    break

##
def plot_batch(batch_imgs, batch_seg):
    batch_size = len(batch_imgs)
    fig, axs = plt.subplots(batch_size, 2, figsize=(10,5))
    print(batch_seg.shape)
    for i in range(batch_size):
        axs[i,0].imshow(batch_imgs[i,0,:,:])
        axs[i,0].set_title(f"Image {i} from batch")
        axs[i,1].imshow(batch_seg[i,0,:,:])
        axs[i,1].set_title(f"Seg {i} from batch")
    plt.show()

## ------- Custom dataset ------
class MyDataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.imgs_names = os.listdir(img_dir)
        self.imgs_len = len(self.imgs_names)
        self.transform = transform

    def __len__(self):
        return self.imgs_len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs_names[idx])
        seg_path = os.path.join(self.labels_dir, self.imgs_names[idx])
        image = read_image(img_path)
        seg = read_image(seg_path)

        if self.transform:
            image = self.transform(image)
            seg = self.transform(seg)

        return image, seg


## ----- DataLoader --------
root_path = "/home/olmozavala/Dropbox/MyCourses/2023/ISC_4933_5935_DataScience_meets_HealthSciences/Examples/ISC_5935_Examples/PyTorch/data/"
dataset = MyDataset(join(root_path,'imgs'), join(root_path,'labels'))

myloader = DataLoader(dataset, batch_size=2, shuffle=True)

##
for batch in myloader:
    x, y = batch
    plot_batch(x, y)
    print('Batch size:', x.size())
    break
##

