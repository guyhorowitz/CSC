import abc

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn


def log_sum_exp(value, weights=None, dim=None):
    eps = 1e-20
    m, idx = torch.max(value, dim=dim, keepdim=True)
    if weights is not None:
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(value - m) * weights, dim=dim) + eps)
    else:
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(value - m), dim=dim) + eps)


class Kernel(abc.ABC, nn.Module):

    def __init__(self, bandwidth=1.0):
        super().__init__()
        self.bandwidth = bandwidth

    def diffs(self, test_Xs, train_Xs):
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs, sample_weight):
        """Computes log p(x) for each x in test_Xs given train_Xs."""
        raise NotImplemented


class GaussianKernel(Kernel):

    def forward(self, test_X, train_X, sample_weight):
        n, d = train_X.shape
        h = torch.tensor(self.bandwidth)
        if sample_weight is not None:
            sample_weight = sample_weight / torch.sum(sample_weight)
            n = 1
        n = torch.tensor(n, dtype=torch.float32)
        pi = torch.tensor(np.pi)

        Z = 0.5 * d * torch.log(2 * pi) + d * torch.log(h) + torch.log(n)
        diffs = self.diffs(test_X, train_X) / h
        diffs = torch.norm(diffs, p=2, dim=-1) ** 2
        return log_sum_exp(-.5 * diffs - Z, dim=-1, weights=sample_weight)


class KernelDensityEstimator(BaseEstimator):

    def __init__(self, train_X=None, bandwidth=1.0, sample_weight=None):
        self.kernel = GaussianKernel()
        self.train_X = train_X
        self.bandwidth = bandwidth
        self.sample_weight = sample_weight

    def fit(self, X, y=None, sample_weight=None):
        self.train_X = X
        self.sample_weight = sample_weight
        self.kernel.bandwidth = self.bandwidth

    def predict(self, x):
        return self.kernel(x, self.train_X, self.sample_weight)

    def score(self, x):
        return torch.sum(self.predict(x))
