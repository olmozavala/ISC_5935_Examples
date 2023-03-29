import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from MyModels import CNN_Classification, Simple_CNN
import matplotlib.pyplot as plt
from Training import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load the MNIST dataset
data_folder = "/data/TORCH_TEST"
download = True
workers = 10
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# #%%
# data_folder = "/data/Caltech"
# dataset = torchvision.datasets.Caltech101(root=data_folder, train=True, download=True)


#%%
dataset = torchvision.datasets.MNIST(root=data_folder, train=True, download=download, transform=transform)
# test_dataset = torchvision.datasets.MNIST(root=data_folder, train=False, download=download, transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=workers)
print("Done loading data!")

#%% Visualize the data
# Get a batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(images[0].numpy().squeeze(), cmap='gray')
ax.set_title(f'Label: {labels[0]}, shape: {images[0].shape}')
plt.show()

#%% Initialize the model, loss, and optimizer
model = Simple_CNN().to(device)
# model = CNN_Classification(28, 28, num_levels=2, num_filters=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
model = train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device)

#%% Show the results
# Get a batch of test data
plot_n = 2
dataiter = iter(val_loader)
data, target = next(dataiter)
data, target = data.to(device), target.to(device)
output = model(data)
fig, ax = plt.subplots(plot_n, 1, figsize=(5, 5*plot_n))
for i in range(plot_n):
    ax[i].imshow(data[i].to('cpu').numpy().squeeze(), cmap='gray')
    ax[i].set_title(f'True: {target[i]}, Prediction: {output[i].argmax(dim=0)} {[f"{x:0.2f}" for x in output[i]]}', wrap=True)
plt.show()
print("Done!")

#%% Showing wrong labels
ouput_value = output.argmax(dim=1).cpu()
dif = np.where(ouput_value != target.cpu())[0]
cur_idx = 0
#%%
fig, ax = plt.subplots(plot_n, 1, figsize=(5, 5*plot_n))
for i in range(plot_n):
    ax[i].imshow(data[dif[cur_idx]].to('cpu').numpy().squeeze(), cmap='gray')
    ax[i].set_title(f'True: {target[dif[cur_idx]]}, Prediction: {output[dif[cur_idx]].argmax(dim=0)} {[f"{x:0.2f}" for x in output[dif[cur_idx]]]}', wrap=True)
    cur_idx += 1
plt.show()
print("Done!")
##

