import abc
import math
from typing import Optional, Union

import scipy.integrate as integrate
import torch
from sklearn.neighbors import KernelDensity
from torch import nn

from density_estimation import KernelDensityEstimator
from utils import split_vector


class Cost(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, x_tag: torch.Tensor):
        raise NotImplemented


class Delta(abc.ABC):
    def __init__(self, cls: nn.Module, cost: Cost):
        self.cls = cls
        self.cost = cost

    @abc.abstractmethod
    def exact(self, x: torch.Tensor, return_ind: bool = False):
        raise NotImplemented

    @abc.abstractmethod
    def approx(self, x: torch.Tensor, return_ind: bool = False):
        raise NotImplemented


class DeltaInverse(abc.ABC):
    def __init__(self, delta: Delta):
        self.delta = delta

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor):
        raise NotImplemented


class QuadraticCost(Cost):
    def __init__(self, cost_matrix: torch.Tensor, cost_matrix_inv: torch.Tensor):
        self.cost_matrix = cost_matrix
        self.cost_matrix_inv = cost_matrix_inv

    def __call__(self, x: torch.Tensor, x_tag: torch.Tensor):
        batch_matrix = self.cost_matrix.expand(x.shape[0], x.shape[1], x.shape[1])
        batch_vectors = torch.unsqueeze((x - x_tag), 2)
        cost = torch.squeeze(torch.bmm(torch.unsqueeze(x - x_tag, 1), torch.bmm(batch_matrix, batch_vectors)), 2)
        return cost


class QuadraticCostDelta(Delta):
    def __init__(self, cls: Optional[nn.Linear], cost: QuadraticCost, tau: float):
        super().__init__(cls, cost)
        self.tau = tau

    def __calc_step_size(self, x):
        denom = torch.mm(self.cls.weight, torch.mm(self.cost.cost_matrix_inv, self.cls.weight.T))
        step_size = self.cls(x) / denom
        step_size = torch.minimum(torch.zeros_like(step_size), step_size)
        return step_size

    def __exact_ind(self, cost):
        bellow_2_ind = (cost <= 2).float()
        above_0_ind = (cost > 0).float()
        return bellow_2_ind * above_0_ind

    def __approx_ind_for_x_opt(self, cost):
        bellow_2_ind = torch.sigmoid((2 - cost) * self.tau)
        return bellow_2_ind

    def __approx_ind_for_h(self, cost):
        bellow_2_ind = torch.sigmoid((2 - cost) * self.tau)
        above_0_ind = (torch.sigmoid(cost * self.tau) - 0.5) * 2
        return bellow_2_ind * above_0_ind

    def __get_x_tag(self, x: torch.Tensor, ind_func_for_x_opt: callable, return_ind_func: callable):
        step_size = self.__calc_step_size(x)  # (N, 1)
        x_tag = x - torch.mm(step_size, torch.mm(self.cost.cost_matrix_inv, self.cls.weight.T).T)
        cost = self.cost(x, x_tag)
        ind_for_x_opt = ind_func_for_x_opt(cost)
        x_tag = x + (x_tag - x) * ind_for_x_opt
        if return_ind_func is not None:
            ind_to_return = return_ind_func(cost)
            return x_tag, ind_to_return
        else:
            return x_tag

    def exact(self, x, return_ind=False):
        return_ind_func = self.__exact_ind if return_ind else None
        x_tag = self.__get_x_tag(x, self.__exact_ind, return_ind_func)
        return x_tag

    def approx(self, x, return_ind_for_h=False):
        return_ind_func = self.__approx_ind_for_h if return_ind_for_h else None
        return self.__get_x_tag(x, self.__approx_ind_for_x_opt, return_ind_func)


class QuadraticCostDeltaInverse(DeltaInverse):
    def __init__(self, delta: QuadraticCostDelta, dens_est: Union[KernelDensity, KernelDensityEstimator], x1_dim: int):
        super().__init__(delta)
        self.dens_est = dens_est
        self.x1_dim = x1_dim
        self.normalized_w = self.__get_normalized_w()
        self.max_t = self.__get_max_t()

    def __get_normalized_w(self):
        w = self.delta.cls.weight.T
        return w / torch.linalg.vector_norm(w)

    def __get_max_t(self):
        A = torch.linalg.cholesky(self.delta.cost.cost_matrix, upper=True)
        return math.sqrt(2) / torch.linalg.vector_norm(torch.mm(A, self.normalized_w))

    def __get_boundary_idx(self, X):
        """
        returns the indices of the x's that on the decision boundary of the classifier
        """
        scores = self.delta.cls(X)
        return (torch.abs(scores) < 1e-6).nonzero()[:, 0]

    def __dens(self, x_tag, t):
        """
        returns the density p(x'-t*w) where w is the weight vector of the classifier
        """
        point = x_tag - t * self.normalized_w.reshape(-1)
        density = torch.exp(self.dens_est.predict((point.reshape(1, -1))))
        return density

    def __get_expected_x(self, x_tag):
        """
        x_tag should be one point
        """
        num = integrate.quad(lambda t: t * self.__dens(x_tag, t), 0, self.max_t, full_output=1)[0]
        denom = integrate.quad(lambda t: self.__dens(x_tag, t), 0, self.max_t, full_output=1)[0]
        # handle the rare case of 0
        if denom == 0:
            return x_tag
        step = num / denom
        x1_tag, x2_tag = split_vector(x_tag.reshape(1, -1), self.x1_dim)
        _, w_x2 = split_vector(self.normalized_w.T, self.x1_dim)
        exp_x2 = x2_tag - w_x2 * step
        exp_x = torch.concat([x1_tag, exp_x2], 1)
        return exp_x

    def __call__(self, X: torch.Tensor):
        inv_X = X.detach().clone()
        bound_idx = self.__get_boundary_idx(X)
        for idx in bound_idx:
            inv_X[idx] = self.__get_expected_x(X[idx]).reshape(1, -1)
        return inv_X


def create_cost_functions(cost_scale: float, x1_dim: int, x2_dim: int):
    n_features = x1_dim + x2_dim

    cost_matrix = torch.eye(n_features) * cost_scale
    cost_matrix_inv = torch.eye(n_features) * (1 / cost_scale)
    cost = QuadraticCost(cost_matrix, cost_matrix_inv)

    X1_cost_matrix = torch.eye(x1_dim) * cost_scale
    X1_cost_matrix_inv = torch.eye(x1_dim) * (1 / cost_scale)
    X1_cost = QuadraticCost(X1_cost_matrix, X1_cost_matrix_inv)

    X2_cost_matrix = torch.eye(x2_dim) * cost_scale
    X2_cost_matrix_inv = torch.eye(x2_dim) * (1 / cost_scale)
    X2_cost = QuadraticCost(X2_cost_matrix, X2_cost_matrix_inv)

    return cost, X1_cost, X2_cost

