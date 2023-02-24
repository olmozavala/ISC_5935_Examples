import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(F"Using {device} device")
#%% 
## Simple tensor
vec_tensor = torch.tensor([1,3,6,4,2,3,3])
vec_tensor_rand = torch.rand(10)
mat_tensor = torch.Tensor(
    [
     [[1, 2], [3, 4]], 
     [[5, 6], [7, 8]], 
     [[9, 0], [1, 2]]
    ]
)

## Properties
print(F"Sum of tensor: {mat_tensor.sum()}")
print(F"Flatten of tensor: {mat_tensor.flatten()}")
print(F"Tensor: {mat_tensor}")
print(F"Tensor Tranpose: {mat_tensor.t}")
print(F"Shape: {mat_tensor.shape}")
print(F"Size = {mat_tensor.size()}")
print(F"Rank = {len(mat_tensor.shape)}")
print(F"Number of elements = {mat_tensor.numel()}")

## GPU
print(F"Initial Device: {mat_tensor.device}")
gpu_tensor = mat_tensor.to(device)
print(F"Initial Device: {gpu_tensor.device}")

## Indexing
print(mat_tensor)
print(F"mat_tensor[1]: {mat_tensor[1]}")
print(F"mat_tensor[1, 1, 0]: {mat_tensor[1, 1, 0]}")
print(F"mat_tensor[1, 1, 0].item(): {mat_tensor[1, 1, 0].item()}")  # Scalar value
print(F"mat_tensor[:, 0, 0]: {mat_tensor[:, 0, 0]}")

## Initialization
print(F"Init ones: {torch.ones_like(mat_tensor)}")
print(F"Init zeros: {torch.zeros_like(mat_tensor)}")
print(F"Init random: {torch.randn_like(mat_tensor)}")
print(F"Init random gpu: {torch.randn(2, 2, device='cuda')}")  # Alternatively 'cuda' or 'cpu'

## Basic Functions
(mat_tensor - 5) * 2
print("Mean:", mat_tensor.mean())
print("Stdev:", mat_tensor.std())
mat_tensor.mean(0)
# Equivalently, you could also write:
# mat_tensor.mean(dim=0)
# mat_tensor.mean(axis=0)
# torch.mean(mat_tensor, 0)
# torch.mean(mat_tensor, dim=0)
# torch.mean(mat_tensor, axis=0)

print(f"Unique vec: {torch.unique(vec_tensor)}")
print(f"Reshape : {mat_tensor.reshape(1,-1)}")  # Reshape as col vector
print(f"Reshape : {mat_tensor.reshape(-1,1)}")  # Reshape as row
##