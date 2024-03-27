import torch.nn as nn
import torch


# Define the CNN architecture
class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, 8, 3, 1, padding='same'))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(8, 8, 3, 1, padding='same'))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Conv2d(8, 8, 3, 1, padding='same'))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(8, 8, 3, 1, padding='same'))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

# ----------------------------------------------------------------------------------
# ----------------------------- Custom size classification CNN -------------------------------------------
# ----------------------------------------------------------------------------------

def level_block(layers, cnn_per_level=2, activation=nn.ReLU, in_filters=8, out_filters=16, kernel_size=3):
    for i in range(cnn_per_level):
        if i == 0:
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size, 1, padding='same'))
        else:
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size, 1, padding='same'))
        layers.append(activation())
    layers.append(nn.MaxPool2d(2))
    return layers

# Define the CNN architecture
class CNN_Classification(nn.Module):

    def __init__(self, in_w, in_h, num_classes=10, cnn_per_level=2, num_levels=3,
                 num_channels=1, num_filters=16, kernel_size=3):
        super(CNN_Classification, self).__init__()
        cur_w = in_w
        cur_h = in_h
        self.layers = nn.ModuleList()
        for c_level in range(num_levels):
            print(f'Level {c_level} in: {cur_w}x{cur_h}x{num_filters}')
            if c_level == 0:
                self.layers = level_block(self.layers, cnn_per_level, nn.ReLU, num_channels, num_filters, kernel_size)
            else:
                self.layers = level_block(self.layers, cnn_per_level, nn.ReLU, num_filters, num_filters, kernel_size)
            cur_w = cur_w // 2
            cur_h = cur_h // 2
            print(f'Level {c_level} out: {cur_w}x{cur_h}x{num_filters}')
        self.fc1 = nn.Linear(num_filters*cur_w*cur_h, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        for c_layer in self.layers:
            x = c_layer(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = nn.functional.log_softmax(self.fc2(x), dim=1)
        return x
