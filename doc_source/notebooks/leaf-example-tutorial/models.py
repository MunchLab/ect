import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#KERNEL_SIZE = 3
KERNEL_SIZE = 3
pad_size = math.floor(KERNEL_SIZE/2)

class CNN(nn.Module): 
    def __init__(self, num_classes, num_channels):
        super(CNN, self).__init__()
        self.padC = [pad_size,pad_size,0,0]
        self.padZ = [0,0,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=10, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=KERNEL_SIZE)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.min_pool2d(self.conv1(x), 2))
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = F.relu(F.min_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

#Same as CNN model, but with zero padding everywhere instead of cylinder padding
class CNN_images(nn.Module): 
    def __init__(self, num_classes):
        super(CNN_images, self).__init__()
        self.padZ = [pad_size,pad_size,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=KERNEL_SIZE)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1920, 1024)
        #self.fc1 = nn.Linear(9680, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.pad(input=x, pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #added additional pad for consistency 
        x = F.pad(input=x, pad=self.padZ, mode='constant')
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


class CNN_simple(nn.Module): 
    def __init__(self, num_classes, num_channels):
        super(CNN_simple, self).__init__()
        self.padC = [pad_size,pad_size,0,0]
        self.padZ = [0,0,pad_size,pad_size]
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=4, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=KERNEL_SIZE)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1540, 1024)
        #self.fc2 = nn.Linear(1024, num_classes)
        self.fc2 = nn.Linear(576, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.shape)
        x = F.pad(F.pad(input=x, pad=self.padC, mode='circular'), pad=self.padZ, mode='constant')
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x    