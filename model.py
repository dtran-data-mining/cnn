'''
requisite modules for model
'''
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    '''
    CNN Model for MNIST Classification

    Structure:
        - Input layer
            - 1 channel (grayscale color)
        - Convolutional layer 1
            - Batch normalization
            - Dropout
            - Max pooling
        - Convolutional layer 2
            - Batch normalization
            - Dropout
            - Max pooling
        - Fully connected layer
            - 3 sequential linear nodes
            - Dropout for all 3 nodes
        - Output layer
            - 10 channels (with classification probabilities for 10 digits, 0-9)


    Parameters:
        - dout_p: dropout probability
        - channel_out1: num output channels for convolutional layer 1
        - channel_out2: num output channels for convolutional layer 2
        - fc_hidden1: num output channels for hidden node 1 of fully connected layer
        - fc_hidden2: num output channels for hidden node 2 of fully connected layer
        - out_size: num of output channels (10 default for MNIST dataset)
    '''

    def __init__(
        self,
        dout_p=0.5,
        channel_out1=64,
        channel_out2=64,
        fc_hidden1=100,
        fc_hidden2=100,
        out_size=10
    ):
        super(CNNModel, self).__init__()
        # convolutional layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=channel_out1,
                kernel_size=5),
            nn.BatchNorm2d(channel_out1),
            nn.ReLU(),
            nn.Dropout(p=dout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # convolutional layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_out1,
                out_channels=channel_out2,
                kernel_size=5),
            nn.BatchNorm2d(channel_out2),
            nn.ReLU(),
            nn.Dropout(p=dout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected layer with 3 sequential nodes
        self.fc1 = nn.Sequential(
            nn.Linear(channel_out2 * 4 * 4, fc_hidden1),
            nn.ReLU(),
            nn.Dropout(p=dout_p),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.ReLU(),
            nn.Dropout(p=dout_p),
            nn.Linear(fc_hidden2, out_size)
        )

    def forward(self, x):
        '''feed forward data through network'''
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the convolution results to (batch_size, n)
        x = torch.flatten(x, 1)
        y_out = self.fc1(x)
        return y_out
