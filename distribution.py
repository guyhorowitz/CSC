import abc
from typing import List, Tuple, Union

import torch

from delta import Delta
from utils import split_vector, estimate_density_sklearn, get_preds


class Distribution(abc.ABC):
    def __init__(self, x1_dim):
        self.x1_dim = x1_dim

    @abc.abstractmethod
    def sample(self, n_samples):
        raise NotImplemented


class CustomSyntheticDataDistribution2(Distribution):
    def __init__(self, x1_dim: int, u_to_x2_func: callable):
        super().__init__(x1_dim)
        self.u_to_x2_func = u_to_x2_func

    def sample(self, n_samples):
        neg_cluster_center = (-3, -3)
        pos_cluster_center = (0, 0)
        std = 0.3
        X1_pos = torch.randn(n_samples // 2, 1) * std + pos_cluster_center[0]
        U_pos = torch.randn(n_samples // 2, 1) * std + pos_cluster_center[1]
        X1_neg = torch.randn(n_samples // 2, 1) * std + neg_cluster_center[0]
        U_neg = torch.randn(n_samples // 2, 1) * std * 1.5 + neg_cluster_center[1]

        U = torch.concat([U_pos, U_neg], 0)
        X1 = torch.concat([X1_pos, X1_neg], 0)
        X2 = self.u_to_x2_func(U)
        X_U = torch.concat([X1, X2, U], 1)
        # shuffle rows
        X_U = X_U[torch.randperm(X_U.size()[0])]
        return X_U


class SyntheticDistribution2D(Distribution):
    def __init__(self, clusters_centers: List[Tuple], u_to_x2_func: callable, std: Union[float, List[float]] = 0.3, clusters_weights: List[float] = None):
        super().__init__(x1_dim=1)
        self.clusters_centers = clusters_centers
        self.n_clusters = len(self.clusters_centers)
        self.clusters_weights = clusters_weights if clusters_weights else [1 / self.n_clusters] * self.n_clusters
        self.stds = std if isinstance(std, list) else [std] * self.n_clusters
        self.u_to_x2_func = u_to_x2_func

    def sample(self, n_samples):
        X1_list = [torch.randn(int(n_samples * weight), 1) * std + center[0] for center, weight, std in zip(self.clusters_centers, self.clusters_weights, self.stds)]
        U_list = [torch.randn(int(n_samples * weight), 1) * std + center[1] for center, weight, std in zip(self.clusters_centers, self.clusters_weights, self.stds)]
        U = torch.concat(U_list, 0)
        X1 = torch.concat(X1_list, 0)
        X2 = self.u_to_x2_func(U)
        X_U = torch.concat([X1, X2, U], 1)
        # shuffle rows
        X_U = X_U[torch.randperm(X_U.size()[0])]
        return X_U


class RealPositiveAndGenerativeNegetiveDistribution(Distribution):
    def __init__(self, x1_dim: int, u_to_x2_func: callable, load_data_func: callable, seed: int, h_star):
        super().__init__(x1_dim)
        self.x1_dim = x1_dim
        self.u_to_x2_func = u_to_x2_func
        self.h_star = h_star
        X1_U, Y = load_data_func(seed)
        self.pos_dist = SequentialDistribution(x1_dim, X1_U[Y == 1])
        self.neg_dist = estimate_density_sklearn(X1_U[Y == -1])

    def sample(self, n_samples):
        n_pos, n_neg = 0, 0
        pos_tensors_list, neg_tensors_list = [], []
        while (n_pos < n_samples // 2) or (n_neg < n_samples // 2):
            if n_pos < n_samples // 2:
                X1_U_pos = torch.FloatTensor(self.pos_dist.sample(n_samples // 2))
                Y = get_preds(self.h_star(X1_U_pos)).view(-1)
                X1_U_pos = X1_U_pos[Y == 1]
                n_pos_remaining = (n_samples // 2) - n_pos
                if len(X1_U_pos) > n_pos_remaining:
                    X1_U_pos = X1_U_pos[:n_pos_remaining]
                n_pos += len(X1_U_pos)
                pos_tensors_list.append(X1_U_pos)
            if n_neg < n_samples // 2:
                X1_U_neg = torch.FloatTensor(self.neg_dist.sample(n_samples // 2))
                Y = get_preds(self.h_star(X1_U_neg)).view(-1)
                X1_U_neg = X1_U_neg[Y == -1]
                n_neg_remaining = (n_samples // 2) - n_neg
                if len(X1_U_neg) > n_neg_remaining:
                    X1_U_neg = X1_U_neg[:n_neg_remaining]
                n_neg += len(X1_U_neg)
                neg_tensors_list.append(X1_U_neg)

        X1_U_pos = torch.vstack(pos_tensors_list)
        X1_U_neg = torch.vstack(neg_tensors_list)
        X1_U = torch.concat([X1_U_pos, X1_U_neg], 0)
        # shuffle rows
        X1_U = X1_U[torch.randperm(X1_U.size()[0])]
        X1, U = split_vector(X1_U, self.x1_dim)
        X2 = self.u_to_x2_func(U)
        X_U = torch.concat([X1, X2, U], 1)
        return X_U


class SequentialDistribution(Distribution):
    def __init__(self, x1_dim: int, X: torch.Tensor):
        super().__init__(x1_dim)
        self.data = X
        self.idx = 0

    def sample(self, n_samples):
        X = self.data[self.idx:self.idx + n_samples]
        self.idx += n_samples
        return X


class InducedDistribution(Distribution):
    def __init__(self, base_dist: Distribution, delta: Delta, x_dim: int):
        super().__init__(base_dist.x1_dim)
        self.base_dist = base_dist
        self.delta = delta
        self.x_dim = x_dim

    def sample(self, n_samples):
        X_U = self.base_dist.sample(n_samples)
        X, U = split_vector(X_U, self.x_dim)
        X_opt = self.delta.exact(X)
        return torch.concat([X_opt, U], 1)

    def sample_only_x1(self, n_samples):
        X_U = self.base_dist.sample(n_samples)
        X, U = split_vector(X_U, self.x_dim)
        X1, _ = split_vector(X, self.x1_dim)
        X_opt = self.delta.exact(X1)
        return torch.concat([X_opt, U], 1)

