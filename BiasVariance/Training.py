import torch
import torch.nn as nn
from MyModels import DenseModel

def training(x, y, x_val, y_val, optimizer, loss=nn.MSELoss(), model = DenseModel(), epochs=500, device='cuda'):
    X = torch.reshape(x, (x.shape[0], 1)).to(device)
    Y = torch.reshape(y, (y.shape[0], 1)).to(device)

    X_val = torch.reshape(x_val, (x_val.shape[0], 1)).to(device)
    Y_val = torch.reshape(y_val, (y_val.shape[0], 1)).to(device)
    model.train()
    loss_history = []
    loss_history_val = []
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        pred_val = model(X_val)
        c_loss = loss(pred, Y)
        c_loss_val = loss(pred_val, Y_val)
        # Backpropagation
        c_loss.backward()
        optimizer.step()

        loss_history.append(c_loss.item())
        loss_history_val.append(c_loss_val.item())

    return loss_history, loss_history_val, model
