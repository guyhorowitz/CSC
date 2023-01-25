from itertools import combinations
from typing import List, Tuple

import torch
from dataclasses import dataclass


@dataclass
class SensitivityParams:
    perm: torch.Tensor
    perm_inv: torch.Tensor
    true_x1_dim: int


def generate_permutations_for_feature_type_flips(x1_dim, x2_dim, n_flips) -> (torch.Tensor, int):
    """
    :param x1_dim:
    :param x2_dim:
    :param n_flips:
    :return: (permutation, assumed_x1_dim)
    """
    n_features = x1_dim + x2_dim
    subsets = [list(subset) for subset in combinations(range(n_features), n_flips)]
    res = []
    for subset in subsets:
        new_x1_features = [*filter(lambda x: x >= x1_dim, subset)]
        new_x2_features = [*filter(lambda x: x < x1_dim, subset)]
        x1_dim_increase = len(new_x1_features) - len(new_x2_features)
        new_x1_dim = x1_dim + x1_dim_increase
        unchanged_features = list(set(range(n_features)) - set(subset))
        perm = new_x1_features + unchanged_features + new_x2_features
        res.append((perm, new_x1_dim))
    return res


def generate_partitions_for_feature_drops(x1_features, x2_features, u_features, n_drops) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    :param x1_dim:
    :param x2_dim:
    :param n_drops:
    :return: list of (x1_features, x2_features, u_features) tuples
    """
    x_features = x1_features + x2_features
    subsets = [list(subset) for subset in combinations(x_features, n_drops)]
    res = []
    for subset in subsets:
        x1_dropped_features = list(set(x1_features).intersection(set(subset)))
        x2_dropped_features = list(set(x2_features).intersection(set(subset)))
        # move x1_dropped_features from x1 to the end of u
        cur_x1_features = list(set(x1_features) - set(x1_dropped_features))
        cur_u_features = u_features + x1_dropped_features
        # remove x2_dropped_features from x2 and move them to the end of u
        cur_x2_features = list(set(x2_features) - set(x2_dropped_features))
        u_without_x2_features = list(set(cur_u_features) - set(x2_features))
        cur_u_features = cur_x2_features + u_without_x2_features + x2_dropped_features
        res.append((cur_x1_features, cur_x2_features, cur_u_features))
    return res

