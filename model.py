"""
define moduals of model
"""
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    """docstring for ClassName"""

    def __init__(self, fc1, fc2, fc3):
        super(CNNModel, self).__init__()
        # write the code of model architecture
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

    def forward(self, x):
        # feed input features to the specified models above
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
