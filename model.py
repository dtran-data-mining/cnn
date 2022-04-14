'''
define moduals of model
'''
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    '''docstring for ClassName'''

    def __init__(self, dout_p=0.5, in_size=100, fc_hidden1=100, fc_hidden2=100, out_size=10):
        super(CNNModel, self).__init__()
        # write the code of model architecture
        self.mlp = nn.Sequential(nn.Conv2d(in_size, fc_hidden1, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.Dropout(p=dout_p),
                                 nn.Conv2d(fc_hidden1, fc_hidden2,
                                           kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.Dropout(p=dout_p),
                                 nn.Linear(fc_hidden2, out_size))

    def forward(self, x):
        y_out = self.mlp(x)
        return y_out
