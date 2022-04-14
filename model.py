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
        self.fc1 = nn.Sequential(nn.Linear(in_size, fc_hidden1),
                                 nn.ReLU(),
                                 nn.Dropout(p=dout_p),
                                 nn.Linear(fc_hidden1, fc_hidden2),
                                 nn.ReLU(),
                                 nn.Dropout(p=dout_p),
                                 nn.Linear(fc_hidden2, out_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_out = self.fc1(x)
        return y_out
