import math

import torch
from torch import nn

from utils import transform_Y_to_zero_one, local_seed


def hinge_loss(Y, Y_pred, mean=True):
    loss = torch.clamp(1 - Y_pred * Y, min=0)
    if mean:
        loss = torch.mean(loss)
    return loss


def weighted_hinge_loss(preds, probs):
    loss = probs * hinge_loss(preds, torch.ones_like(preds), False) + (1 - probs) * hinge_loss(preds, -1 * torch.ones_like(preds), False)
    return torch.mean(loss)


class CrossEntropy:
    def __init__(self, neg_w):
        self.neg_w = neg_w

    def __call__(self, preds, target):
        weights = (target == 1).float() + self.neg_w * (target == -1).float()
        ce = nn.BCEWithLogitsLoss(weights)
        # target is in {-1, 1}. this loss requires {0, 1}
        Y = transform_Y_to_zero_one(target)
        return ce(preds, Y)


def cross_entropy_loss_for_h(preds, target):
    ce = nn.BCEWithLogitsLoss()
    # target is in {-1, 1}. this loss requires {0, 1}
    Y = transform_Y_to_zero_one(target)
    return ce(preds, Y)


def s_hinge_loss(X: torch.Tensor, y: torch.Tensor, f: nn.Linear, cost_scale: float):
    W = f.weight[0]
    max_cost = math.sqrt(2 / cost_scale)
    return torch.mean(torch.relu(1 - y * f(X) - max_cost * y * torch.norm(W, p=2)))


class PolynomialModel(nn.Module):
    def __init__(self, x_dim, degrees_list):
        super().__init__()
        self._degrees_list = degrees_list
        self.linear = nn.Linear(len(self._degrees_list) * x_dim, 1)

    def forward(self, x):
        return self.linear(self._polynomial_features(x))

    def _polynomial_features(self, x):
        return torch.cat([x ** i for i in self._degrees_list], 1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers):
        super().__init__()
        self.hidden_layers = hidden_layers

        self.hidden = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.hidden.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)


class SklearnModel(nn.Module):
    def __init__(self, sklearn_model):
        super(SklearnModel, self).__init__()
        self.sklearn_model = sklearn_model

    def forward(self, x: torch.Tensor):
        #out_numpy = self.sklearn_model.score_samples(x.detach().numpy())
        out_numpy = self.sklearn_model.predict(x.detach().numpy())
        return torch.reshape(torch.from_numpy(out_numpy), (-1, 1)).float()


class TrickyFeatureModel:
    def __init__(self, h_star, bad_feature, threshold, gt=True, slope=1):
        self.h_star = h_star
        self.bad_feature = bad_feature
        self.threshold = threshold
        self.gt = gt
        self.slope = slope

    def __call__(self, X):
        y = self.h_star(X).view(-1)
        if self.gt:
            dist_from_threshold_in_bad_feature = X[:, self.bad_feature] - self.threshold
        else:
            dist_from_threshold_in_bad_feature = self.threshold - X[:, self.bad_feature]
        y += torch.relu(dist_from_threshold_in_bad_feature) * -self.slope
        # print stats
        n_above_threshold = (dist_from_threshold_in_bad_feature > 0).float().mean()
        mean_dist_from_threshold = dist_from_threshold_in_bad_feature[dist_from_threshold_in_bad_feature > 0].mean()
        print(f"TrickyFeature: points above threshold: {n_above_threshold}, mean dist from threshold of these points: {mean_dist_from_threshold}")
        return y.view(-1, 1)


class StochasticModel:
    def __init__(self, h_star, max_flip_prob, slope):
        if not 0 <= max_flip_prob <= 1:
            raise ValueError(f"max_flip_prob = {max_flip_prob}. it should be 0 <= max_flip_prob <= 1")
        if slope <= 0:
            raise ValueError(f"slope = {slope}. it should be 0 < slop")
        self.h_star = h_star
        self.max_flip_prob = max_flip_prob
        self.slope = slope

    def flip_scores(self, scores):
        new_scores = scores.detach().clone()
        with local_seed(0):
            probs_for_flip = torch.relu(self.max_flip_prob - self.slope * new_scores.abs())
            flip_mask = torch.bernoulli(probs_for_flip).bool()
            # flip scores
            new_scores[flip_mask] *= -1
        return new_scores

    def __call__(self, X):
        scores = self.h_star(X)
        new_scores = self.flip_scores(scores)
        # print stats
        flip_rate = (scores != new_scores).float().mean().item()
        print(f"StochasticModel: {round(flip_rate * 100, 2)}% flips")
        return new_scores


class HelperModel:
    def __init__(self, model: callable, temp: torch.nn.Parameter = nn.Parameter(torch.ones(1))):
        self.model = model
        self.temp = temp

