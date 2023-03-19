import torch
import torch.nn as nn
from MyModels import DenseModel

def training(x, y, optimizer, loss=nn.MSELoss(), model = DenseModel(), epochs=500, device='cuda'):
    X = torch.reshape(x, (x.shape[0], 1)).to(device)
    Y = torch.reshape(y, (y.shape[0], 1)).to(device)
    model.train()
    loss_history = []
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        c_loss = loss(pred, Y)
        # Backpropagation
        c_loss.backward()
        optimizer.step()

        loss_history.append(c_loss.item())

    return loss_history, model
