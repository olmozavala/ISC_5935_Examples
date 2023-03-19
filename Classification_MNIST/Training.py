import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    cur_time = datetime.now()
    writer = SummaryWriter(f'logs/MNIST_{cur_time.strftime("%Y%m%d-%H%M%S")}')
    for epoch in range(num_epochs):
        model.train()
        # Loop over each batch from the training set (to update the model)
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f'{65*batch_idx}/{len(train_loader.dataset)}', end='\r')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        val_loss = 0
        correct = 0
        # Loop over each batch from the test set (to evaluate the model)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', correct / len(val_loader.dataset), epoch)

        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)

        print(f'Epoch: {epoch+1}, Val loss: {val_loss:.4f}, Val Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.2f}%)')

    print("Done!")
    writer.close()
    return model
#%%

