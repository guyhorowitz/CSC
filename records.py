import itertools
from typing import List

import pandas as pd
import torch
from dataclasses import dataclass, field

from delta import Delta
from testing import test_strategic, test_strategic_only_X2_model
from utils import split_vector, get_preds, HiddenPrints, Data, transform_Y_to_zero_one


@dataclass
class AccuracyRecord:
    train_acc: float = field(default=None)
    val_acc: float = field(default=None)
    test_acc: float = field(default=None)


@dataclass
class ExpResults:
    real_acc_record: AccuracyRecord = field(default_factory=AccuracyRecord)
    perceived_acc_record: AccuracyRecord = field(default_factory=AccuracyRecord)
    test_movement_ind: torch.Tensor = field(default_factory=AccuracyRecord)
    test_pre_move_y: torch.Tensor = field(default=None)
    test_post_move_y: torch.Tensor = field(default=None)
    test_pre_move_preds: torch.Tensor = field(default=None)
    test_post_move_preds: torch.Tensor = field(default=None)
    test_costs: torch.Tensor = field(default=None)
    test_dists: torch.Tensor = field(default=None)

    def collect_strategic_results(self, model: torch.nn.Module, h_star: callable, delta: Delta, data: Data, x1_dim: int):
        with HiddenPrints():
            # accuracies
            self.real_acc_record.train_acc = test_strategic(model, h_star, delta, data.X_train, data.U_train, x1_dim, verbose=False)
            self.real_acc_record.val_acc = test_strategic(model, h_star, delta, data.X_val, data.U_val, x1_dim, verbose=False)
            self.real_acc_record.test_acc = test_strategic(model, h_star, delta, data.X_test, data.U_test, x1_dim, verbose=False)
            # pre move data
            X1, _ = split_vector(data.X_test, x1_dim)
            self.test_pre_move_y = get_preds(h_star(torch.concat([X1, data.U_test], 1)))
            self.test_pre_move_preds = get_preds(model(data.X_test))
            # post move data
            X_opt, movement_ind = delta.exact(data.X_test, True)
            self.test_movement_ind = movement_ind
            self.test_costs = delta.cost(data.X_test, X_opt)
            self.test_dists = torch.linalg.norm(data.X_test - X_opt, dim=1).view(-1, 1)
            X1_opt, _ = split_vector(X_opt, x1_dim)
            self.test_post_move_y = get_preds(h_star(torch.concat([X1_opt, data.U_test], 1)))
            self.test_post_move_preds = get_preds(model(X_opt))

    def collect_strategic_results_using_only_x2(self, model: torch.nn.Module, h_star: callable, delta: Delta, data: Data, x1_dim: int):
        with HiddenPrints():
            X1_train, X2_train = split_vector(data.X_train, x1_dim)
            X1_val, X2_val = split_vector(data.X_val, x1_dim)
            X1_test, X2_test = split_vector(data.X_test, x1_dim)
            # accuracies
            self.real_acc_record.train_acc = test_strategic_only_X2_model(model, h_star, delta, X1_train, X2_train, data.U_train)
            self.real_acc_record.val_acc = test_strategic_only_X2_model(model, h_star, delta, X1_val, X2_val, data.U_val)
            self.real_acc_record.test_acc = test_strategic_only_X2_model(model, h_star, delta, X1_test, X2_test, data.U_test)
            # pre move data
            self.test_pre_move_y = get_preds(h_star(torch.concat([X1_test, data.U_test], 1)))
            self.test_pre_move_preds = get_preds(model(X2_test))
            # post move data
            X2_opt_test, movement_ind = delta.exact(X2_test, True)
            self.test_movement_ind = movement_ind
            self.test_costs = delta.cost(X2_test, X2_opt_test)
            self.test_dists = torch.linalg.norm(X2_test - X2_opt_test, dim=1).view(-1, 1)
            # X1_opt, _ = split_vector(X_opt, x1_dim)
            self.test_post_move_y = get_preds(h_star(torch.concat([X1_test, data.U_test], 1)))
            self.test_post_move_preds = get_preds(model(X2_opt_test))

    def calc_stats(self, method: str):
        data = {}
        data['accuracy_train'] = [self.real_acc_record.train_acc]
        data['accuracy_val'] = [self.real_acc_record.val_acc]
        data['accuracy_test'] = [self.real_acc_record.test_acc]
        data['perceived_accuracy_train'] = [self.perceived_acc_record.train_acc]
        data['perceived_accuracy_val'] = [self.perceived_acc_record.val_acc]
        data['perceived_accuracy_test'] = [self.perceived_acc_record.test_acc]
        Y = transform_Y_to_zero_one(self.test_pre_move_y)
        Y_tag = transform_Y_to_zero_one(self.test_post_move_y)
        pre_move_preds = transform_Y_to_zero_one(self.test_pre_move_preds)
        post_move_preds = transform_Y_to_zero_one(self.test_post_move_preds)
        for move, y, y_tag, pre_move_pred, post_move_pred in itertools.product(*([(0, 1)] * 5)):
            cond_stats = (self.test_movement_ind == move) & (Y == y) & (Y_tag == y_tag) & (pre_move_preds == pre_move_pred) & (post_move_preds == post_move_pred)
            data[f'move {move} y {y} ytag {y_tag} pre_pred {pre_move_pred} post_pred {post_move_pred}'] = [cond_stats.float().mean().item()]
        data['mean_utility'] = [(self.test_post_move_preds - self.test_costs).mean().item()]
        data['mean_burden y 1'] = [(self.test_costs[Y == 1]).mean().item()]
        data['mean_burden y 1 ytag 1'] = [(self.test_costs[(Y == 1) & (Y_tag == 1)]).mean().item()]
        data['mean_distance'] = [self.test_dists.mean().item()]
        data['mean_distance move 1'] = [(self.test_dists[self.test_movement_ind == 1]).mean().item()]
        data['mean_cost'] = [self.test_costs.mean().item()]
        data['mean_cost move 1'] = [self.test_costs[self.test_movement_ind == 1].mean().item()]
        df = pd.DataFrame(data=data)
        df["method"] = method
        return df


def results_list_to_multiple_rounds_stats(res_list: List[ExpResults], method: str) -> pd.DataFrame:
    stats_list = list(map(lambda res: res.calc_stats(method), res_list))
    for i, stats in enumerate(stats_list):
        stats["round"] = i + 1
    return pd.concat(stats_list).reset_index(drop=True)


def results_list_to_chosen_model_stats(res_list: List[ExpResults], method: str) -> pd.DataFrame:
    chosen_model_results = max(res_list, key=lambda res: res.real_acc_record.val_acc)
    return chosen_model_results.calc_stats(method)


def get_chosen_model_and_multiple_rounds_stats(res_list: List[ExpResults], method: str) -> (pd.DataFrame, pd.DataFrame):
    return results_list_to_chosen_model_stats(res_list, method), results_list_to_multiple_rounds_stats(res_list, method)

