import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from models import LogisticRegression, weights_init_normal, test_model
from FairRobustSampler import FairRobust, CustomDataset

from argparse import Namespace

import warnings

warnings.filterwarnings("ignore")


def lr_epoch(model, train_features, labels, optimizer):
    """Trains the model with the given train data.

    Args:
        model: A torch model to train.
        train_features: A torch tensor indicating the train features.
        labels: A torch tensor indicating the true labels.
        optimizer: A torch optimizer.

    Returns:
        loss values.
    """

    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    loss = nn.BCELoss()((F.tanh(label_predicted.squeeze()) + 1) / 2, (labels.squeeze() + 1) / 2)
    loss.backward()

    optimizer.step()

    return loss.item()


def svm_epoch(model, train_features, labels, optimizer):
    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    loss = torch.mean(torch.clamp(1 - labels.squeeze() * label_predicted.squeeze(), min=0))
    loss.backward()

    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    n_samples = 3000
    train_size = 0.66
    n_epochs = 400
    poi_ratio = 0.1
    seed = 1

    algorithm = 'lr'

    test = False

    if test:
        n_features = 3
        xz_train = np.load('./synthetic_data/xz_train.npy')
        y_train = np.load('./synthetic_data/y_noise_general.npy')
        z_train = np.load('./synthetic_data/z_train.npy')

        xz_test = np.load('./synthetic_data/xz_test.npy')
        y_test = np.load('./synthetic_data/y_test.npy')
        z_test = np.load('./synthetic_data/z_test.npy')

        z_train = torch.FloatTensor(z_train)
        z_test = torch.FloatTensor(z_test)
    else:
        n_features = 20
        x, y = make_classification(n_samples, n_features - 1)
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
        z = np.zeros(n_samples)
        z[:int(n_samples / 2)] = 1
        random.shuffle(z)
        xz = np.hstack((x, z[:, np.newaxis]))
        y_noise = np.copy(y)
        for i in range(len(y_noise)):
            if random.random() < poi_ratio:
                if y_noise[i] == -1:
                    y_noise[i] = 1
                else:
                    y_noise[i] = -1

        xz_train, xz_test, y_train, y_test = train_test_split(xz, y_noise, train_size=train_size)

        z_train = torch.FloatTensor(xz_train[:, -1])
        z_test = torch.FloatTensor(xz_test[:, -1])

    xz_train = torch.FloatTensor(xz_train)
    y_train = torch.FloatTensor(y_train)

    xz_test = torch.FloatTensor(xz_test)
    y_test = torch.FloatTensor(y_test)

    if torch.cuda.is_available():
        xz_train = xz_train.cuda()
        y_train = y_train.cuda()
        z_train = z_train.cuda()

        xz_test = xz_test.cuda()
        y_test = y_test.cuda()
        z_test = z_test.cuda()

    print("---------- Number of Data ----------")
    print(
        "Train data : %d, Test data : %d "
        % (len(y_train), len(y_test))
    )
    print("------------------------------------")

    full_tests = []

    parameters = Namespace(warm_start=100, tau=1 - poi_ratio, alpha=0.001, batch_size=100)
    train_data = CustomDataset(xz_train, y_train, z_train)

    model = nn.Linear(n_features, 1)
    if torch.cuda.is_available():
        model.cuda()
    if seed is not None:
        torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    losses = []

    sampler = FairRobust(model, train_data.x, train_data.y, train_data.z, target_fairness='eqodds',
                         algorithm=algorithm, parameters=parameters, replacement=False, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)

    for epoch in range(n_epochs):
        print(epoch, end="\r")

        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate(train_loader):
            if algorithm == 'lr':
                loss = lr_epoch(model, data, target, optimizer)
            elif algorithm == 'svm':
                loss = svm_epoch(model, data, target, optimizer)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss) / len(tmp_loss))

    tmp_test = test_model(model, xz_test, y_test, z_test)
    print("  Test accuracy: {}, EO disparity: {}".format(tmp_test['Acc'], tmp_test['EqOdds_diff']))
