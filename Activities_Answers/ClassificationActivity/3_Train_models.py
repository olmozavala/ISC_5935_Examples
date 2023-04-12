# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import autocontrast, equalize
# Local libraries
from proj_ai.Training import train_model
from proj_ai.Generators import ClassificationDataset
from os.path import join
from models.MyModels import CNN_Classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Using device: ", device)

#%% ----- DataLoader --------
# ssh_folder = "/data/LOCAL_GOFFISH/AVISO/"
data_folder = "/home/olmozavala/Dropbox/MyCourses/2023/ISC_4933_5935_DataScience_meets_HealthSciences/DataSets/Chest_X-Ray_small/chest_xray/"
train_folder = join(data_folder, "train")
val_folder = join(data_folder, "val")

# img_size = (512, 728)
img_size = (256, 384)
transforms = transforms.Compose([
    equalize,  # Equalize the histogram of the input image (first)
    autocontrast,
    transforms.ToTensor(),
    transforms.Resize((img_size[0], img_size[1])),
])

train_dataset = ClassificationDataset(train_folder, transform=transforms)
val_dataset = ClassificationDataset(val_folder, transform=transforms)

print("Total number of training samples: ", len(train_dataset))
print("Total number of validation samples: ", len(val_dataset))

# Create DataLoaders for training and validation
workers = 5
# train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True, num_workers=workers)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True, num_workers=workers)
print("Done loading data!")

#%% Initialize the model, loss, and optimizer
model = CNN_Classification(img_size[0], img_size[1], num_levels=2, num_channels=1, num_filters=8).to(device)

loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
model = train_model(model, optimizer, loss_func, train_loader, val_loader, num_epochs, device=device, output_folder='OUTPUT')