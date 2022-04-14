'''
Fun: CNN for MNIST classification
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# import queue
# from util import _create_batch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNNModel


# input hyper-paras
parser = argparse.ArgumentParser(description='neural networks')
parser.add_argument('-mode', dest='mode', type=str,
                    default='train', help='train or test')
parser.add_argument('-num_epoches', dest='num_epoches',
                    type=int, default=40, help='num of epoches')

parser.add_argument('-fc_hidden1', dest='fc_hidden1', type=int,
                    default=100, help='dim of hidden neurons')
parser.add_argument('-fc_hidden2', dest='fc_hidden2', type=int,
                    default=100, help='dim of hidden neurons')
parser.add_argument('-learning_rate', dest='learning_rate',
                    type=float, default=0.01, help='learning rate')
parser.add_argument('-decay', dest='decay', type=float,
                    default=0.5, help='learning rate')
parser.add_argument('-batch_size', dest='batch_size',
                    type=int, default=100, help='batch size')
parser.add_argument('-rotation', dest='rotation',
                    type=int, default=10, help='transform random rotation')
parser.add_argument('-dropout', dest='dropout', type=float,
                    default=0.4, help='dropout prob')

parser.add_argument('-activation', dest='activation', type=str,
                    default='relu', help='activation function')
parser.add_argument('-MC', dest='MC', type=int, default=10,
                    help='number of monte carlo')
parser.add_argument('-channel_out1', dest='channel_out1',
                    type=int, default=64, help='number of channels')
parser.add_argument('-channel_out2', dest='channel_out2',
                    type=int, default=64, help='number of channels')
parser.add_argument('-k_size', dest='k_size', type=int,
                    default=4, help='size of filter')
parser.add_argument('-pooling_size', dest='pooling_size',
                    type=int, default=2, help='size for max pooling')
parser.add_argument('-stride', dest='stride', type=int,
                    default=1, help='stride for filter')
parser.add_argument('-max_stride', dest='max_stride',
                    type=int, default=2, help='stride for max pooling')
parser.add_argument('-ckp_path', dest='ckp_path', type=str,
                    default='checkpoint', help='path of checkpoint')

args = parser.parse_args()


# -------------------------------------------------------
# data loader
# -------------------------------------------------------
def _load_data(data_path, batch_size):
    # training data loader
    train_trans = transforms.Compose([transforms.RandomRotation(args.rotation), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.MNIST(
        root=data_path, download=False, train=True, transform=train_trans)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # testing data loader
    test_trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = torchvision.datasets.MNIST(
        root=data_path, download=False, train=False, transform=test_trans)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def _save_checkpoint(ckp_path, model, optimizer, epoches, global_step):
    # save checkpoint to ckp_path: 'checkpoint/step_{global_step}.pt'
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoches,
                  'global_step': global_step}

    torch.save(checkpoint, ckp_path)


def _load_checkpoint(ckp_path, model, optimizer, epoches, global_step):
    # load checkpoint from ckp_path='checkpoint/step_100.pt'
    checkpoint = torch.load(ckp_path)

    # load parameters (W and b) to models
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoches = checkpoint['epoch']
    global_step = checkpoint['global_step']


def _compute_accuracy(model, test_loader):
    correct_pred = {classname: 0 for classname in range(9)}
    total_pred = {classname: 0 for classname in range(9)}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, y_labels = data
            outputs = model(images)
            _, y_preds = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for y_label, y_pred in zip(y_labels, y_preds):
                if y_label == y_pred:
                    correct_pred[y_label] += 1
                total_pred[y_label] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    return correct_pred, total_pred


def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    '''Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs'''
    decay_t = 0
    if (epoch > 5):
        decay_t += 1
    if (epoch >= 10):
        decay_t += 1
    if (epoch > 20):
        decay_t += 1
    learning_rate *= decay ** decay_t

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('learning_rate: ', learning_rate)


def _test_model(model, device, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for _, (x_batch, y_labels) in enumerate(test_loader):
            x_batch, y_labels = Variable(x_batch).to(
                device), Variable(y_labels).to(device)
            y_out = model(x_batch)
            _, y_preds = torch.max(y_out.data, 1)
            correct += (y_preds == y_labels).sum().item()
            total += y_labels.size(0)

    return correct / total * 100


def main():
    # -------------------------------------------------------
    # decide to use gpu or cpu
    # -------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed(72)

    # -------------------------------------------------------
    # model initialization
    # -------------------------------------------------------
    model = CNNModel(dout_p=args.dropout, in_size=28*28,
                     fc_hidden1=args.fc_hidden1, fc_hidden2=args.fc_hidden2, out_size=10)
    model.to(device)

    # ----------------------------------------
    # optimizer and loss function
    # ----------------------------------------
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # cross entropy loss
    loss_fn = nn.CrossEntropyLoss()

    # ----------------------------------------
    # model training
    # ----------------------------------------
    writer = SummaryWriter()
    train_loader, test_loader = _load_data('data', args.batch_size)

    if args.mode == 'train':
        epoches = args.num_epoches
        global_step = 0
        # load checkpoint
        _load_checkpoint(args.ckp_path, model, optimizer, epoches, global_step)

        model = model.train()

        for epoch in range(epoches):
            adjust_learning_rate(args.learning_rate,
                                 optimizer, epoch, args.decay)

            for _, (x_batch, y_labels) in enumerate(train_loader):
                global_step += 1
                x_batch, y_labels = Variable(x_batch).to(
                    device), Variable(y_labels).to(device)

                # train model
                y_preds = model(x_batch)
                loss = loss_fn(y_preds, y_labels)

                # back prop
                optimizer.zero_grad()
                loss.backward()
                # update params
                optimizer.step()

                # save checkpoint
                if global_step % 100 == 0:
                    _save_checkpoint(args.ckp_path, model,
                                     optimizer, epoches, global_step)

    # ------------------------------------
    # model testing
    # ------------------------------------
    elif args.mode == 'test':
        model.eval()
        accuracy = _test_model(model, device, test_loader)
        print(f'accuracy: {accuracy}%')

    else:
        print(f'invalid mode: {args.mode}, please enter \'train\' or \'test\'')


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('running time: ', (time_end - time_start)/60.0, 'mins')
