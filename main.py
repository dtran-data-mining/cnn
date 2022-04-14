"""
Fun: CNN for CIFAR10 classification
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import argparse
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# import queue
# from util import _create_batch
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNNModel


# input hyper-paras
parser = argparse.ArgumentParser(description="neural networks")
parser.add_argument("-mode", dest="mode", type=str,
                    default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches",
                    type=int, default=40, help="num of epoches")

parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int,
                    default=100, help="dim of hidden neurons")
parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int,
                    default=100, help="dim of hidden neurons")
parser.add_argument("-learning_rate", dest="learning_rate",
                    type=float, default=0.01, help="learning rate")
parser.add_argument("-decay", dest="decay", type=float,
                    default=0.5, help="learning rate")
parser.add_argument("-batch_size", dest="batch_size",
                    type=int, default=100, help="batch size")
parser.add_argument("-rotation", dest="rotation",
                    type=int, default=10, help="transform random rotation")
parser.add_argument("-dropout", dest="dropout", type=float,
                    default=0.4, help="dropout prob")

parser.add_argument("-activation", dest="activation", type=str,
                    default='relu', help="activation function")
parser.add_argument("-MC", dest='MC', type=int, default=10,
                    help="number of monte carlo")
parser.add_argument("-channel_out1", dest='channel_out1',
                    type=int, default=64, help="number of channels")
parser.add_argument("-channel_out2", dest='channel_out2',
                    type=int, default=64, help="number of channels")
parser.add_argument("-k_size", dest='k_size', type=int,
                    default=4, help="size of filter")
parser.add_argument("-pooling_size", dest='pooling_size',
                    type=int, default=2, help="size for max pooling")
parser.add_argument("-stride", dest='stride', type=int,
                    default=1, help="stride for filter")
parser.add_argument("-max_stride", dest='max_stride',
                    type=int, default=2, help="stride for max pooling")
parser.add_argument("-ckp_path", dest='ckp_path', type=str,
                    default="checkpoint", help="path of checkpoint")

args = parser.parse_args()


def _load_data(DATA_PATH, batch_size):
    '''Data loader'''

    print("data_path: ", DATA_PATH)
    train_trans = transforms.Compose([transforms.RandomRotation(args.rotation), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH, download=True, train=True, transform=train_trans)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # for testing
    test_trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH, download=False, train=False, transform=test_trans)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def _compute_accuracy(args):
    ## please write the code below ##
    return


def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
    lr = learning_rate
    if (epoch > 5):
        lr = 0.001
    if (epoch >= 10):
        lr = 0.0001
    if (epoch > 20):
        lr = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print("learning_rate: ", lr)


def _test_model(model):
    ## please write the code below ##
    return


def main():
    use_cuda = torch.cuda.is_available()  # if have gpu or cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(72)

    # -------------------------------------------------------
    # please write the code about model initialization below
    # -------------------------------------------------------
    fc1 = nn.Linear(16 * 5 * 5, 120)
    fc2 = nn.Linear(120, 84)
    fc3 = nn.Linear(84, 10)
    model = CNNModel(fc1, fc2, fc3)
    model.to(device)

    # ---------------------------------------------------
    ### please write the Optimizer and LOSS FUNCTION ##
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9)  # optimizer
    loss_fun = nn.CrossEntropyLoss()  # cross entropy loss

    # ----------------------------------------
    # model training code below
    # ----------------------------------------
    BATCH_SIZE = 100
    train_loader, test_loader = _load_data("data", BATCH_SIZE)

# ------------------------------------
# model testing code below
# ------------------------------------
# model.eval()
# accy_naive = _test_model()


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
