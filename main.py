'''
CNN for MNIST Classification

Contributors: Duke Tran, Abdi Hassan
Date: 4/20/22
CSCI 420 - Data Mining
'''

import time
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
# tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CNNModel


# input hyper-parameters
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
                    default=0.1, help='learning rate')
parser.add_argument('-batch_size', dest='batch_size',
                    type=int, default=100, help='batch size')
parser.add_argument('-rotation', dest='rotation',
                    type=int, default=10, help='transform random rotation')
parser.add_argument('-dropout', dest='dropout', type=float,
                    default=0.5, help='dropout prob')

parser.add_argument('-activation', dest='activation', type=str,
                    default='relu', help='activation function')
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

MNIST_SIZE = 60000


def _load_data(data_path, batch_size):
    '''loads in MNIST dataset, prepares DataLoaders for training and testing'''
    # training data loader
    train_trans = transforms.Compose([transforms.RandomRotation(args.rotation), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # set download to True on initial run
    train_dataset = torchvision.datasets.MNIST(
        root=data_path, download=False, train=True, transform=train_trans)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # testing data loader
    test_trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # set download to True on initial run
    test_dataset = torchvision.datasets.MNIST(
        root=data_path, download=False, train=False, transform=test_trans)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def _save_checkpoint(ckp_path, model, optimizer, params):
    '''saves checkpoint to ckp_path: 'checkpoint/step_{global_step}.pt, stores parameters'''
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': params['epoch'],
                  'global_step': params['global_step'],
                  'learning_rate': params['learning_rate']}

    torch.save(checkpoint, ckp_path)


def _load_checkpoint(ckp_path, model, optimizer, params):
    '''loads checkpoint from ckp_path='checkpoint/step_{global_step}.pt, loads in parameters'''
    checkpoint = torch.load(ckp_path)

    # load parameters (W and b) to models
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    params['epoch'] = checkpoint['epoch']
    params['global_step'] = checkpoint['global_step']
    params['learning_rate'] = checkpoint['learning_rate']


def _compute_accuracy(y_preds, y_labels):
    '''computes accuracy for y predictions of given batch compared to y labels'''
    return (y_preds == y_labels).sum().item()


def _compute_class_accuracy(model, test_loader):
    '''computes accuracy for each class (digits 0-9)'''
    correct_preds = {num_class: 0 for num_class in range(10)}
    total_preds = {num_class: 0 for num_class in range(10)}

    # no gradient needed
    with torch.no_grad():
        for data in test_loader:
            x_data, y_labels = data
            outputs = model(x_data)
            _, y_preds = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for y_pred, y_label in zip(y_preds, y_labels):
                if y_pred == y_label:
                    correct_preds[y_label.item()] += 1
                total_preds[y_label.item()] += 1

    # print accuracy for each class
    for classname, correct_count in correct_preds.items():
        accuracy = 100 * float(correct_count) / total_preds[classname]
        print(f'accuracy for class \'{classname}\': {accuracy:.3f}%')

    return correct_preds, total_preds


def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    '''sets the learning rate to the initial LR decayed by 1/8 every args.lr epochs'''
    decay_t = 0
    if (epoch > 5):
        decay_t += 1
    if (epoch > 10):
        decay_t += 1
    if (epoch > 20):
        decay_t += 1
    # stop decaying after epoch 30
    if (epoch < 30):
        learning_rate *= decay ** decay_t

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('learning_rate: ', learning_rate)


def _test_model(model, device, test_loader):
    '''tests model against dataset, returns accuracy percentage'''
    correct, total = 0, 0
    with torch.no_grad():
        for _, (x_batch, y_labels) in enumerate(test_loader):
            x_batch, y_labels = Variable(x_batch).to(
                device), Variable(y_labels).to(device)
            y_out = model(x_batch)
            _, y_preds = torch.max(y_out.data, 1)
            correct += _compute_accuracy(y_preds, y_labels)
            total += y_labels.size(0)

    return correct / total * 100


def main():
    '''sets up script and device, loads in data, trains/tests'''
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
    model = CNNModel(
        dout_p=args.dropout,
        channel_out1=args.channel_out1,
        channel_out2=args.channel_out2,
        fc_hidden1=args.fc_hidden1,
        fc_hidden2=args.fc_hidden2,
        out_size=10
    )
    model.to(device)

    # ----------------------------------------
    # optimizer and loss function
    # ----------------------------------------
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # cross entropy loss
    loss_fn = nn.CrossEntropyLoss()

    train_loader, test_loader = _load_data('data', args.batch_size)

    writer = SummaryWriter()
    writer.close()

    params = {
        'epoch': 0,
        'global_step': 0,
        'learning_rate': args.learning_rate
    }

    # ----------------------------------------
    # model training
    # ----------------------------------------
    if args.mode == 'train':
        model.train()

        # load checkpoint (only if checkpoint file exists and is populated)
        _load_checkpoint(args.ckp_path, model, optimizer, params)

        for epoch in range(params['epoch'], args.num_epoches):
            params['epoch'] = epoch
            adjust_learning_rate(params['learning_rate'],
                                 optimizer, epoch, args.decay)

            sum_loss = 0
            for _, (x_batch, y_labels) in enumerate(train_loader):
                params['global_step'] += 1
                x_batch, y_labels = Variable(x_batch).to(
                    device), Variable(y_labels).to(device)

                # train model
                y_preds = model(x_batch)
                loss = loss_fn(y_preds, y_labels)
                sum_loss += loss

                # back prop
                optimizer.zero_grad()
                loss.backward()
                # update params
                optimizer.step()

                # save checkpoint
                if params['global_step'] % 100 == 0:
                    _save_checkpoint(args.ckp_path, model, optimizer, params)

            # plot loss using tensorboard
            writer.add_scalar('loss/train', sum_loss /
                              (MNIST_SIZE / args.batch_size), epoch)

    # ------------------------------------
    # model testing
    # ------------------------------------
    elif args.mode == 'test':
        model.eval()

        _load_checkpoint(args.ckp_path, model, optimizer, params)

        accuracy = _test_model(model, device, test_loader)
        print(f'testing accuracy: {accuracy:.3f}%')

        # test classification accuracy for each class
        # _compute_class_accuracy(model, test_loader)

    else:
        print(f'invalid mode: {args.mode}, please enter \'train\' or \'test\'')


if __name__ == '__main__':
    '''module entry point'''
    time_start = time.time()
    main()
    time_end = time.time()
    print(f'running time: {((time_end - time_start)/60.0):.5f} mins')
