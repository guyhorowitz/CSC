import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
from dataclasses import dataclass, field
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from density_estimation import KernelDensityEstimator


@dataclass
class ExplorationParams:
    should_use: bool
    coef: float
    decay: float


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@contextmanager
def local_seed(seed):
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def split_data(X, Y, percentage, shuffle=False):
    if shuffle:
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        Y = Y[perm]
    num_val = int(len(X) * percentage)
    return X[num_val:], Y[num_val:], X[:num_val], Y[:num_val]


def split_data_3(X, U, Y, percentage):
    num_val = int(len(X) * percentage)
    return X[num_val:], U[num_val:], Y[num_val:], X[:num_val], U[:num_val], Y[:num_val]


def transform_Y_to_zero_one(Y: torch.Tensor):
    new_Y = Y.detach().clone()
    new_Y[new_Y == -1] = 0
    return new_Y


def shuffle(X, Y):
    torch.manual_seed(0)
    np.random.seed(0)
    data = torch.cat((Y, X), 1)
    data = data[torch.randperm(data.size()[0])]
    X = data[:, 1:]
    Y = data[:, 0]
    return X, Y


def get_movement_percent(movement_ind: torch.Tensor):
    return movement_ind.mean()


def calc_accuracy(Y, Ypred):
    Y_pred = torch.sign(Ypred)
    Y_pred[Y_pred == 0] = 1
    num = len(Y)
    temp = Y.view(-1) - Y_pred.view(-1)
    acc = len(temp[temp == 0]) * 1. / num
    return acc


def calc_accuracy_with_movement(Y, Ypred, movement_ind):
    Y_pred = torch.sign(Ypred)
    Y_pred[Y_pred == 0] = 1
    # x that moves should be classify as 1
    Y_pred[movement_ind == 1] = 1
    num = len(Y)
    temp = Y.view(-1) - Y_pred.view(-1)
    acc = len(temp[temp == 0]) * 1. / num
    return acc


def split_vector(X, idx):
    X_1, X_2 = X[:, :idx], X[:, idx:]
    if len(X_1.shape) == 1:
        X_1 = torch.unsqueeze(X_1, 1)
    if len(X_2.shape) == 1:
        X_2 = torch.unsqueeze(X_2, 1)
    return X_1, X_2


def estimate_density(estimator_cls, X, weights=None):
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(estimator_cls(), params)
    grid.fit(X, sample_weight=weights)
    return grid.best_estimator_


def estimate_density_torch(X, weights=None):
    return estimate_density(KernelDensityEstimator, X, weights)


def estimate_density_sklearn(X, weights=None):
    return estimate_density(KernelDensity, X, weights)


def create_Y(X, U, h_star, x1_dim):
    X1, _ = split_vector(X, x1_dim)
    X1_U = torch.concat([X1, U], 1)
    with torch.no_grad():
        Y = get_preds(h_star(X1_U))
    return Y


def create_uniform_samples(low, high, steps, dim, x1_dim, u_to_x2_func, h_star):
    axes = [torch.linspace(low, high, steps=steps) for _ in range(dim)]
    X1_U = torch.cartesian_prod(*axes)
    with torch.no_grad():
        Y = get_preds(h_star(X1_U))
    X1, U = split_vector(X1_U, x1_dim)
    X2 = u_to_x2_func(U)
    X_U = torch.concat([X1, X2, U], 1)
    return X_U, Y


def get_preds(scores):
    preds = torch.sign(scores)
    preds[preds == 0] = 1
    return preds


def get_time_string():
    return datetime.now().strftime("%d_%m_%Y-%H_%M_%S")


@dataclass
class Data:
    X_train: torch.Tensor = field(default=None)
    Y_train: torch.Tensor = field(default=None)
    U_train: torch.Tensor = field(default=None)
    X_val: torch.Tensor = field(default=None)
    Y_val: torch.Tensor = field(default=None)
    U_val: torch.Tensor = field(default=None)
    X_test: torch.Tensor = field(default=None)
    Y_test: torch.Tensor = field(default=None)
    U_test: torch.Tensor = field(default=None)

