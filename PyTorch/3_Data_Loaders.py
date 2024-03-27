# %% This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

##
def plot_batch(batch_imgs, batch_seg):
    batch_size = len(batch_imgs)
    fig, axs = plt.subplots(batch_size, 2, figsize=(10,5))
    print(f"Batch size when plotting: {batch_seg.shape}")
    for i in range(batch_size):
        axs[i,0].imshow(batch_imgs[i,0,:,:])
        axs[i,0].set_title(f"Image {i} from batch")
        axs[i,1].imshow(batch_seg[i,0,:,:])
        axs[i,1].set_title(f"Seg {i} from batch")
    plt.show()

# %% ------- Custom dataset ------
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
        print(image.shape)

        if self.transform:
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            seg = self.transform(seg)
            # image = self.transform(image)
            # seg = self.transform(seg)

        return image, seg

# %% ----- Transforms --------
transform_pipeline = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.CenterCrop(50),
    # transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
])

root_path = "/home/olmozavala/Dropbox/MyCourses/2023/ISC_4933_5935_DataScience_meets_HealthSciences/Examples/ISC_5935_Examples/PyTorch/data/"
# dataset = MyDataset(join(root_path,'imgs'), join(root_path,'labels'))
dataset = MyDataset(join(root_path,'imgs'), join(root_path,'labels'), 
                    transform=transform_pipeline)

# To modify in a DataLoader: num_workers, batch_size, shuffle
myloader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in myloader:
    x, y = batch
    plot_batch(x, y)
    print('Batch size:', x.size())
    break

# %% Example for spliting the data into train and test from the dataset
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

