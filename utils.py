import math
from random import shuffle, random

import numpy as np
from scipy.stats import multivariate_normal


def gen_gaussian(n_samples, mean_in, cov_in, class_label):
    nv = multivariate_normal(mean=mean_in, cov=cov_in)
    X = nv.rvs(n_samples)
    y = np.ones(n_samples, dtype=float) * class_label
    return nv, X, y


def gen_syn_data(n_samples, mu1, sigma1, mu2, sigma2, phi):
    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1)  # negative class

    # join the positive and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0, n_samples * 2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]

    # Generate the sensitive feature here
    rotation_mult = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    X_aux = np.dot(X, rotation_mult)

    z = []  # this array holds the sensitive feature value
    for i in range(0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        r = np.random.uniform()  # generate a random number from 0 to 1
        if r < p1 / (p1 + p2):  # the first cluster is the positive class
            z.append(1.0)
        else:
            z.append(0.0)

    z = np.array(z)
    xz = np.vstack((X[:, 0], X[:, 1], z)).transpose()

    return xz, y, z


def corrupt_labels(y, poi_ratio=0.1):
    y_noise = np.copy(y)
    for i in range(len(y_noise)):
        if random() < poi_ratio:
            if y_noise[i] == -1:
                y_noise[i] = 1
            else:
                y_noise[i] = -1
    return y_noise


if __name__ == '__main__':
    n_samples = 1000
    phi = math.pi / 2.5  # this variable determines the initial discrimination in the data
    poi_ratio = 0.1  # ratio of artificially corrupted data

    mu1, sigma1 = [2, 2], [[7, 1], [2, 5]]
    mu2, sigma2 = [-2, -2], [[2, 1], [1, 10]]

    gen_syn_data(n_samples, mu1, mu2, sigma1, sigma2, phi)
