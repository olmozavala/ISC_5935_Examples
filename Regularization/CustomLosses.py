import torch.nn as nn
import torch as T

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, outputs, targets):
        loss = T.mean(T.square(outputs - targets))
        return loss

class L1RegularizedMSELoss(nn.Module):
    def __init__(self, l1_lambda):
        super(L1RegularizedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_lambda = l1_lambda

    def forward(self, outputs, targets, model):
        mse_loss = self.mse_loss(outputs, targets)
        l1_loss = 0.0
        for param in model.parameters():
            l1_loss += T.linalg.norm(param, 1)
        return mse_loss + self.l1_lambda * l1_loss

class L2RegularizedMSELoss(nn.Module):
    def __init__(self, l2_lambda):
        super(L2RegularizedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l2_lambda = l2_lambda

    def forward(self, outputs, targets, model):
        mse_loss = self.mse_loss(outputs, targets)
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += T.linalg.norm(param, 2)**2
        return mse_loss + self.l2_lambda * l2_loss

#%% main
if __name__ == '__main__':
    # ------------- Testing losses ------------------
    x = T.tensor([1.0,2.0])
    y = T.tensor([3.0,2.0])
    print(f"{T.norm(x, 1):.4f}")
    mse = nn.MSELoss()
    custom_mse = CustomMSELoss()
    print(f"Torch MSE: {mse(x, y):.4f}")
    print(f"Custom MSE: {custom_mse(x, y):.4f}")


#%%

