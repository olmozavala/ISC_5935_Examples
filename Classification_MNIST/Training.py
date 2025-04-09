import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    cur_time = datetime.now()
    writer = SummaryWriter(f'logs/MNIST_{cur_time.strftime("%Y%m%d-%H%M%S")}')
    
    # Track training loss within each epoch
    train_losses = []
    
    # Prit the total number of examples in the training set
    print(f'Total number of examples in the training set: {len(train_loader.dataset)}')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f'Epoch {epoch+1} of {num_epochs}')
        # Loop over each batch from the training set (to update the model)
        for batch_idx, (data, target) in enumerate(train_loader):
            # Print the size of the batch
            print(f'Batch size: {data.size(0)}')
            print(f'{65*batch_idx}/{len(train_loader.dataset)}', end='\r')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Accumulate training loss
            running_loss += loss.item()
            
            # Log training loss more frequently (every 100 batches)
            if batch_idx % 100 == 99:
                writer.add_scalar('Loss/train_step', 
                                running_loss/100, 
                                epoch * len(train_loader) + batch_idx)
                running_loss = 0.0

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
        
        # Log epoch-level metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', correct / len(val_loader.dataset), epoch)

        # Add model graph and sample images only in the first epoch
        if epoch == 0:
            # Log sample images
            images, labels = next(iter(train_loader))
            images, labels = images.to(device), labels.to(device)
            grid = torchvision.utils.make_grid(images)
            writer.add_image('Sample Training Images', grid, 0)
            
            # Log model graph
            try:
                writer.add_graph(model, images)
            except Exception as e:
                print(f"Failed to add model graph to TensorBoard: {e}")

        print(f'Epoch: {epoch+1}, Val loss: {val_loss:.4f}, Val Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.2f}%)')

    print("Done!")
    writer.close()
    return model