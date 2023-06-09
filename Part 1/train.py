from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch


from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy
import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, data_loader, optimizer, criterion, config):
    # TODO set model to train mode
    model = model.cuda()
    acc = []
    ll = []
    for epoch in range(config.max_epoch):
        correct = 0
        total = 0
        loss = 0
        for _, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Add more code here ...

            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_norm)
            batch_inputs = batch_inputs.cuda()
            batch_targets = batch_targets.cuda()
            outputs = model(batch_inputs)
            optimizer.zero_grad()
            l = criterion(outputs, batch_targets)
            l.backward(retain_graph=True)
            optimizer.step()
            loss += l.item()
            _, ans = torch.max(outputs.data, dim=1)
            total += batch_targets.size(0)
            correct += (ans == batch_targets).sum().item()
        ll.append(loss / len(data_loader))
        acc.append(correct / total)
        optimizer.zero_grad()
    plt.plot(config.max_epoch, ll, label='train_loss')
    plt.plot(config.max_epoch, acc, label="train_acc")
    plt.legend()
    plt.title(f'seq_length={config.input_length}, train')
    plt.show()



@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    model = model.eval()
    model = model.to(device)
    loss = 0
    correct = 0
    total = 0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        if len(batch_targets) < config.batch_size:
            break
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        outputs = model(batch_inputs)

        loss += criterion(outputs, batch_targets).item()

        _, ans = torch.max(outputs.data, 1)
        total += batch_targets.size(0)
        correct += (ans == batch_targets).sum().item()

        if step % 10 == 1:
            print(f'[{step}/{len(data_loader)}]', loss / step, correct / total)
    return loss / len(data_loader), correct / total


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize the model that we are going to use
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                 config.batch_size)  # fixme
    model.to(device)

    # Initialize the dataset and data loader
    # dataset =   # fixme
    # Split dataset into train and validation sets
    train_dataset = PalindromeDataset(config.input_length, int(config.data_size * config.portion_train))  # fixme
    val_dataset = PalindromeDataset(config.input_length, int(config.data_size * (1 - config.portion_train)))
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, config.batch_size)  # fixme
    val_dloader = DataLoader(val_dataset, config.batch_size)  # fixme

    # Set up the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme
    # scheduler = ...  # 不知道这是什么


        # Train the model for one epoch
    train(model, train_dloader, optimizer, criterion, config)

        # Evaluate the trained model on the validation set
    evaluate( model, val_dloader, criterion, device, config)

    print('Done training.')


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
