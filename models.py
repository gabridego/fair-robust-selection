import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def lr_epoch(model, train_features, labels, optimizer):
    """Trains a logistic regression with the given train data.

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


def svm_epoch(model, c, train_features, labels, optimizer):
    """Trains a support vector machines with the given train data.

    Args:
        model: A torch model to train.
        c: Regularization parameter. The strength of the regularization is inversely proportional to C.
           Must be strictly positive. The penalty is a squared l2 penalty.
        train_features: A torch tensor indicating the train features.
        labels: A torch tensor indicating the true labels.
        optimizer: A torch optimizer.

    Returns:
        loss values.
    """

    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    weight = model.weight.squeeze()
    loss = torch.mean(torch.clamp(1 - labels.squeeze() * label_predicted.squeeze(), min=0))
    loss += c * (weight.t() @ weight) / 2.0
    loss.backward()

    optimizer.step()

    return loss.item()


def weights_init_normal(m):
    """Initializes the weight and bias of the model.

    Args:
        m: A torch model to initialize.

    Returns:
        None.
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        nn.init.constant_(m.bias.data, 0)


def test_model(model_, X, y, s1):
    """Tests the performance of a model.

    Args:
        model_: A model to test.
        X: Input features of test data.
        y: True label (1-D) of test data.
        s1: Sensitive attribute (1-D) of test data.

    Returns:
        The test accuracy and the fairness metrics of the model.
    """

    model_.eval()

    y_hat = model_(X).squeeze()
    prediction = (y_hat > 0.0).int().squeeze()
    y = (y > 0.0).int()

    z_0_mask = (s1 == 0.0)
    z_1_mask = (s1 == 1.0)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))

    y_0_mask = (y == 0.0)
    y_1_mask = (y == 1.0)
    y_0 = int(torch.sum(y_0_mask))
    y_1 = int(torch.sum(y_1_mask))

    Pr_y_hat_1 = float(torch.sum((prediction == 1))) / (z_0 + z_1)

    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1

    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))

    Pr_y_hat_1_y_0 = float(torch.sum((prediction == 1)[y_0_mask])) / y_0
    Pr_y_hat_1_y_1 = float(torch.sum((prediction == 1)[y_1_mask])) / y_1

    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1

    y_0_z_0_mask = (y == 0.0) & (s1 == 0.0)
    y_0_z_1_mask = (y == 0.0) & (s1 == 1.0)
    y_0_z_0 = int(torch.sum(y_0_z_0_mask))
    y_0_z_1 = int(torch.sum(y_0_z_1_mask))

    Pr_y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask])) / y_0_z_0
    Pr_y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask])) / y_0_z_1

    recall = Pr_y_hat_1_y_1
    precision = float(torch.sum((prediction == 1)[y_1_mask])) / (int(torch.sum(prediction == 1)) + 0.00001)

    y_hat_neq_y = float(torch.sum((prediction == y.int())))

    test_acc = torch.sum(prediction == y.int()).float() / len(y)
    test_f1 = 2 * recall * precision / (recall + precision + 0.00001)

    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1) + 0.00001
    min_eo_0 = min(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    max_eo_0 = max(Pr_y_hat_1_y_0_z_0, Pr_y_hat_1_y_0_z_1) + 0.00001
    min_eo_1 = min(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001
    max_eo_1 = max(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1) + 0.00001

    DP = max(abs(Pr_y_hat_1_z_0 - Pr_y_hat_1), abs(Pr_y_hat_1_z_1 - Pr_y_hat_1))

    EO_Y_0 = max(abs(Pr_y_hat_1_y_0_z_0 - Pr_y_hat_1_y_0), abs(Pr_y_hat_1_y_0_z_1 - Pr_y_hat_1_y_0))
    EO_Y_1 = max(abs(Pr_y_hat_1_y_1_z_0 - Pr_y_hat_1_y_1), abs(Pr_y_hat_1_y_1_z_1 - Pr_y_hat_1_y_1))

    return {'Acc': test_acc.item(), 'DP_diff': DP, 'EO_Y0_diff': EO_Y_0, 'EO_Y1_diff': EO_Y_1,
            'EqOdds_diff': max(EO_Y_0, EO_Y_1)}


def plot_boundaries(model, xz, y, s1):
    xz = xz.cpu().numpy()
    y = y.cpu().numpy()
    s1 = s1.cpu().numpy()

    x = xz[:, :-1]
    if x.shape[-1] != 2:
        warnings.warn('data do not have 2 features, not plotting')
        return

    delta = 0.001

    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    x_lin = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    plt.figure(figsize=(10, 10))
    plt.xlim([x[:, 0].min() + delta, x[:, 0].max() - delta])
    plt.ylim([x[:, 1].min() + delta, x[:, 1].max() - delta])
    plt.plot(x_lin, (-W[0] / W[1]) * x_lin + (-b / W[1]), c='black', lw=2)
    y_0_z_0 = np.logical_and(y <= 0, s1 == 0)
    y_0_z_1 = np.logical_and(y <= 0, s1 == 1)
    y_1_z_0 = np.logical_and(y > 0, s1 == 0)
    y_1_z_1 = np.logical_and(y > 0, s1 == 1)

    plt.scatter(x[y_1_z_0, 0], x[y_1_z_0, 1], c='blue', marker='x', label="z=0 (pos)")
    plt.scatter(x[y_0_z_0, 0], x[y_0_z_0, 1], c='red', marker='x', label="z=0 (neg)")
    plt.scatter(x[y_1_z_1, 0], x[y_1_z_1, 1], edgecolors='blue', facecolors='none', label="z=1 (pos)")
    plt.scatter(x[y_0_z_1, 0], x[y_0_z_1, 1], edgecolors='red', facecolors='none', label="z=1 (neg)")
    plt.legend()
    plt.show()
