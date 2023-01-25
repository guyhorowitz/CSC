from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from delta import QuadraticCostDelta, QuadraticCostDeltaInverse
from density_estimation import KernelDensityEstimator
from distribution import Distribution, InducedDistribution
from model import HelperModel, s_hinge_loss
from records import ExpResults
from sensitivity_exp_utils import SensitivityParams
from testing import test, test_strategic
from utils import calc_accuracy, calc_accuracy_with_movement, split_data, split_vector, transform_Y_to_zero_one, ExplorationParams, \
    estimate_density_torch, get_preds, HiddenPrints, Data


class NonStrategicTrainer:
    def __init__(self, f, train_dl: DataLoader, X_val, Y_val, X_test, Y_test, opt, loss):
        self.f = f
        self.train_dl = train_dl
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.opt = opt
        self.loss = loss
        self.results = ExpResults()

    def test(self):
        return test(self.f, self.X_test, self.Y_test)

    def train(self, epochs: int, verbose: bool = False, early_stop: int = 7):
        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []

        best_val_error = 1
        consecutive_no_improvement = 0

        for epoch in range(epochs):
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in self.train_dl:
                self.opt.zero_grad()
                Ybatch_pred = self.f(Xbatch)
                l = self.loss(Ybatch_pred.view(-1), Ybatch.view(-1))
                l.backward()
                self.opt.step()
                train_losses[-1].append(l.item())
                with torch.no_grad():
                    e = calc_accuracy(Ybatch, Ybatch_pred)
                    train_errors[-1].append(1 - e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f | err: %3.5f" %
                          (batch, len(self.train_dl), np.mean(train_losses[-1]), np.mean(train_errors[-1])))
                batch += 1

            with torch.no_grad():
                Yval_pred = self.f(self.X_val)
                val_loss = self.loss(Yval_pred.view(-1), self.Y_val.view(-1)).item()
                val_losses.append(val_loss)
                val_error = 1 - calc_accuracy(self.Y_val, Yval_pred)
                val_errors.append(val_error)
                if early_stop:
                    if val_error < best_val_error:
                        consecutive_no_improvement = 0
                        best_val_error = val_error

                    else:
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= early_stop:
                            print(f"trained {epoch} epochs. val acc: {1 - val_error}")
                            break
        if not early_stop:
            print(f"trained {epoch} epochs. val acc: {1 - val_error}")
            # if verbose:
                # print("------------- epoch %03d / %03d | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, val_losses[-1], val_errors[-1]))

    def perceived_test(self, X, Y):
        preds = self.f(X)
        return calc_accuracy(Y, preds)

    def collect_perceived_accs(self, data: Data):
        self.results.perceived_acc_record.train_acc = self.perceived_test(data.X_train, data.Y_train)
        self.results.perceived_acc_record.val_acc = self.perceived_test(data.X_val, data.Y_val)
        self.results.perceived_acc_record.test_acc = self.perceived_test(data.X_test, data.Y_test)

    def collect_trainer_results(self, h_star: callable, delta, data: Data, x1_dim: int):
        self.collect_perceived_accs(data)
        self.results.collect_strategic_results(self.f, h_star, delta, data, x1_dim)
        return self.results


class NonStrategicWithWeightedDatasetsTrainer:
    def __init__(self, f, train_dl_list, val_list, opt, loss):
        """
         train data list should contain tuples of (data, weight)
         val data list should contain tuples of (X, Y, weight)
        """
        self.f = f
        self.train_dl_list = train_dl_list
        self.val_list = val_list
        self.opt = opt
        self.loss = loss

    def train(self, epochs, verbose=False):
        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []

        best_val_error = 1
        consecutive_no_improvement = 0

        for epoch in range(epochs):
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for dl, w in self.train_dl_list:
                for Xbatch, Ybatch in dl:
                    self.opt.zero_grad()
                    Ybatch_pred = self.f(Xbatch)
                    l = self.loss(Ybatch_pred.view(-1), Ybatch.view(-1))
                    l = l * w
                    l.backward()
                    self.opt.step()
                    train_losses[-1].append(l.item())
                    with torch.no_grad():
                        e = calc_accuracy(Ybatch, Ybatch_pred)
                        train_errors[-1].append(1 - e)
                    if verbose:
                        print("batch %03d | loss: %3.5f | err: %3.5f" %
                              (batch, np.mean(train_losses[-1]), np.mean(train_errors[-1])))
                    batch += 1

            with torch.no_grad():
                val_error = 0
                for X_val, Y_val, w in self.val_list:
                    Yval_pred = self.f(X_val)
                    val_loss = self.loss(Yval_pred.view(-1), Y_val.view(-1)).item()
                    val_losses.append(val_loss)
                    val_error += (1 - calc_accuracy(Y_val, Yval_pred)) * w
                val_errors.append(val_error)

                if val_error < best_val_error:
                    consecutive_no_improvement = 0
                    best_val_error = val_error

                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 4:
                        break
            if verbose:
                print("------------- epoch %03d / %03d | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, val_losses[-1], val_errors[-1]))


class StrategicTrainer:
    def __init__(self, f, delta, train_dl, X_val, Y_val, X_test, Y_test, opt, loss, f_reg=0.0, cost_scale=1.0):
        self.f = f
        self.delta = delta
        self.train_dl = train_dl
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.opt = opt
        self.loss = loss
        self.f_reg = f_reg
        self.cost_scale = cost_scale
        self.results = ExpResults()

    def test(self, strategic: bool):
        with torch.no_grad():
            X_opt = self.delta.exact(self.X_test) if strategic else self.X_test
            return test(self.f, X_opt, self.Y_test)

    def init_f_with_non_strategic_training(self, epochs):
        trainer = NonStrategicTrainer(self.f, self.train_dl, self.X_val, self.Y_val, self.X_test, self.Y_test, self.opt, self.loss)
        trainer.train(epochs)

    def train(self, epochs: int, strategic: bool, verbose=False):
        self.init_f_with_non_strategic_training(epochs)
        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []

        best_val_error = 1
        consecutive_no_improvement = 0

        for epoch in range(epochs):
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in self.train_dl:
                self.opt.zero_grad()
                X_opt = self.delta.approx(Xbatch) if strategic else Xbatch
                Ybatch_pred = self.f(X_opt)
                # l = self.loss(Ybatch_pred.view(-1), Ybatch.view(-1))
                l = s_hinge_loss(Xbatch, Ybatch, self.f, self.cost_scale)
                l += self.f_reg * torch.norm(self.f.weight)
                l.backward()
                self.opt.step()
                train_losses[-1].append(l.item())
                with torch.no_grad():
                    e = calc_accuracy(Ybatch, Ybatch_pred)
                    train_errors[-1].append(1 - e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f | err: %3.5f" %
                          (batch, len(self.train_dl), np.mean(train_losses[-1]), np.mean(train_errors[-1])))
                batch += 1

            with torch.no_grad():
                X_opt, movement_ind = self.delta.exact(self.X_val, True)
                Yval_pred = self.f(X_opt)
                val_loss = self.loss(Yval_pred.view(-1), self.Y_val.view(-1)).item()
                val_losses.append(val_loss)
                val_error = 1 - calc_accuracy_with_movement(self.Y_val, Yval_pred, movement_ind)
                val_errors.append(val_error)
                if val_error < best_val_error:
                    consecutive_no_improvement = 0
                    best_val_error = val_error

                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 4:
                        return best_val_error

            if verbose:
                print("------------- epoch %03d / %03d | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, val_losses[-1], val_errors[-1]))

    def perceived_test(self, X, Y, delta):
        X_opt, movement_ind = delta.exact(X, True)
        preds = self.f(X_opt)
        return calc_accuracy_with_movement(Y, preds, movement_ind)

    def collect_perceived_accs(self, data: Data, delta):
        self.results.perceived_acc_record.train_acc = self.perceived_test(data.X_train, data.Y_train, delta)
        self.results.perceived_acc_record.val_acc = self.perceived_test(data.X_val, data.Y_val, delta)
        self.results.perceived_acc_record.test_acc = self.perceived_test(data.X_test, data.Y_test, delta)

    def collect_perceived_accs_using_only_x2(self, data: Data, delta, x1_dim: int):
        _, X2_train = split_vector(data.X_train, x1_dim)
        _, X2_val = split_vector(data.X_val, x1_dim)
        _, X2_test = split_vector(data.X_test, x1_dim)
        self.results.perceived_acc_record.train_acc = self.perceived_test(X2_train, data.Y_train, delta)
        self.results.perceived_acc_record.val_acc = self.perceived_test(X2_val, data.Y_val, delta)
        self.results.perceived_acc_record.test_acc = self.perceived_test(X2_test, data.Y_test, delta)

    def collect_trainer_results(self, h_star: callable, delta, data: Data, x1_dim: int):
        self.collect_perceived_accs(data, delta)
        self.results.collect_strategic_results(self.f, h_star, delta, data, x1_dim)
        return self.results

    def collect_trainer_results_using_only_x2(self, h_star: callable, delta, data: Data, x1_dim: int):
        self.collect_perceived_accs_using_only_x2(data, delta, x1_dim)
        self.results.collect_strategic_results_using_only_x2(self.f, h_star, delta, data, x1_dim)
        return self.results


class CausalStrategicTrainer:
    def __init__(self, f, h: HelperModel, delta, train_dl, X_val, Y_val, X_test, x1_dim, opt, loss, exploration_params: Optional[ExplorationParams],
                 density_estimator: Optional[KernelDensityEstimator] = None, f_reg=0.0, cost_scale=1.0):
        self.f = f
        self.h = h
        self.delta = delta
        self.train_dl = train_dl
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.opt = opt
        self.x1_dim = x1_dim
        self.loss = loss
        self.exploration_params = exploration_params
        self.density_estimator = density_estimator
        self.f_reg = f_reg
        self.cost_scale = cost_scale
        self.results = ExpResults()

    def init_f_with_non_strategic_training(self, epochs):
        trainer = NonStrategicTrainer(self.f, self.train_dl, self.X_val, self.Y_val, self.X_test, None, self.opt, self.loss)
        trainer.train(epochs)

    def get_labels(self, X, X_opt, Y, movement_ind):
        scores = self.get_h_scores(X, X_opt)
        new_labels = get_preds(scores)
        return Y + (new_labels - Y) * movement_ind

    def get_h_scores(self, X, X_opt):
        _, X_2 = split_vector(X, self.x1_dim)
        X_opt1, _ = split_vector(X_opt, self.x1_dim)
        scores = self.h.model(torch.concat([X_opt1, X_2], 1))
        return scores

    def get_labels_for_hinge(self, h_scores, Y, movement_ind):
        h_labels = torch.sigmoid(h_scores) * 2 - 1
        # mu = torch.sigmoid(self.delta.tau * -1 * f_scores_on_original_points) * movement_ind
        # for x that didn't move, the label will be (proxy) of the original label.
        labels = Y + (h_labels - Y) * movement_ind
        return labels

    def density_loss(self, X, X_opt):
        _, X_2 = split_vector(X, self.x1_dim)
        X_opt1, _ = split_vector(X_opt, self.x1_dim)
        density_scores = self.density_estimator.predict(torch.concat([X_opt1, X_2], 1))
        return density_scores.mean()

    def train(self, epochs: int, verbose=False):
        self.init_f_with_non_strategic_training(epochs)

        train_losses = []
        #val_losses = []
        train_errors = []
        val_errors = []

        best_val_error = 1
        consecutive_no_improvement = 0

        for epoch in range(epochs):
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in self.train_dl:
                self.opt.zero_grad()
                X_opt, prox_movement_ind = self.delta.approx(Xbatch, True)
                with torch.no_grad():
                    _, exact_movement_ind = self.delta.exact(Xbatch, True)
                f_scores = self.f(X_opt)
                h_scores = self.get_h_scores(Xbatch, X_opt)
                h_labels = self.get_labels_for_hinge(h_scores, Ybatch, prox_movement_ind)
                l = s_hinge_loss(Xbatch, h_labels, self.f, self.cost_scale)
                l += self.f_reg * torch.norm(self.f.weight)
                if self.exploration_params and self.exploration_params.should_use:
                    l += self.exploration_params.coef * self.density_loss(Xbatch, X_opt)
                l.backward()
                self.opt.step()
                train_losses[-1].append(l.item())
                with torch.no_grad():
                    Y = self.get_labels(Xbatch, X_opt, Ybatch, exact_movement_ind)
                    e = calc_accuracy_with_movement(Y, f_scores, exact_movement_ind)
                    train_errors[-1].append(1 - e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f | err: %3.5f" %
                          (batch, len(self.train_dl), np.mean(train_losses[-1]), np.mean(train_errors[-1])))
                batch += 1

            with torch.no_grad():
                X_opt, movement_ind = self.delta.exact(self.X_val, True)
                Yval_pred = self.f(X_opt)
                Y_val = self.get_labels(self.X_val, X_opt, self.Y_val, movement_ind)
                #val_loss = self.loss(Yval_pred.view(-1), Y_val.view(-1)).item()
                #val_losses.append(val_loss)
                val_error = 1 - calc_accuracy_with_movement(Y_val, Yval_pred, movement_ind)
                val_errors.append(val_error)
                if val_error < best_val_error:
                    consecutive_no_improvement = 0
                    best_val_error = val_error

                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 7:
                        print(f"trained {epoch} epochs. val acc: {1 - val_error}")
                        return best_val_error

            if verbose:
                print("------------- epoch %03d / %03d | err: %3.5f" % (epoch + 1, epochs, val_errors[-1]))
        print(f"trained {epochs} epochs")

    def perceived_test(self, X, Y, delta):
        X_opt, movement_ind = delta.exact(X, True)
        preds = self.f(X_opt)
        Y_tag = self.get_labels(X, X_opt, Y, movement_ind)
        return calc_accuracy_with_movement(Y_tag, preds, movement_ind)

    def collect_perceived_accs(self, data: Data, delta):
        self.results.perceived_acc_record.train_acc = self.perceived_test(data.X_train, data.Y_train, delta)
        self.results.perceived_acc_record.val_acc = self.perceived_test(data.X_val, data.Y_val, delta)
        self.results.perceived_acc_record.test_acc = self.perceived_test(data.X_test, data.Y_test, delta)

    def collect_trainer_results(self, h_star: callable, delta, data: Data, x1_dim: int):
        self.collect_perceived_accs(data, delta)
        self.results.collect_strategic_results(self.f, h_star, delta, data, x1_dim)
        return self.results


class CausalStrategicTrainerWithU(CausalStrategicTrainer):
    def __init__(self, f, h, delta, X_U_train_dl, X_train_dl, X_val, X_Uval, Y_val, X_test, x_dim, x1_dim, opt, loss, f_reg=0, cost_scale=1):
        super().__init__(f, h, delta, X_train_dl, X_val, Y_val, X_test, x1_dim, opt, loss, exploration_params=None, f_reg=f_reg, cost_scale=cost_scale)
        self.x_dim = x_dim
        self.X_U_train_dl = X_U_train_dl
        self.X_Uval = X_Uval

    def get_h_scores(self, X_opt, U):
        X_opt1, _ = split_vector(X_opt, self.x1_dim)
        scores = self.h.model(torch.concat([X_opt1, U], 1))
        return scores

    def get_labels(self, X_opt, U, Y, movement_ind):
        scores = self.get_h_scores(X_opt, U)
        new_labels = get_preds(scores)
        return Y + (new_labels - Y) * movement_ind

    def train(self, epochs: int, verbose=False):
        self.init_f_with_non_strategic_training(epochs)
        train_losses = []
        # val_losses = []
        train_errors = []
        val_errors = []

        best_val_error = 1
        consecutive_no_improvement = 0

        for epoch in range(epochs):
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for X_Ubatch, Ybatch in self.X_U_train_dl:
                self.opt.zero_grad()
                Xbatch, Ubatch = split_vector(X_Ubatch, self.x_dim)
                X_opt, prox_movement_ind = self.delta.approx(Xbatch, True)
                with torch.no_grad():
                    _, exact_movement_ind = self.delta.exact(Xbatch, True)
                f_scores = self.f(X_opt)
                h_scores = self.get_h_scores(X_opt, Ubatch)
                h_labels = self.get_labels_for_hinge(h_scores, Ybatch, prox_movement_ind)
                l = s_hinge_loss(Xbatch, h_labels, self.f, self.cost_scale)
                l += self.f_reg * torch.norm(self.f.weight)
                l.backward()
                self.opt.step()
                train_losses[-1].append(l.item())
                with torch.no_grad():
                    Y = self.get_labels(X_opt, Ubatch, Ybatch, exact_movement_ind)
                    e = calc_accuracy_with_movement(Y, f_scores, exact_movement_ind)
                    train_errors[-1].append(1 - e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f | err: %3.5f" %
                          (batch, len(self.X_U_train_dl), np.mean(train_losses[-1]), np.mean(train_errors[-1])))
                batch += 1

            with torch.no_grad():
                X_val, U_val = split_vector(self.X_Uval, self.x_dim)
                X_opt, movement_ind = self.delta.exact(X_val, True)
                Yval_pred = self.f(X_opt)
                Y_val = self.get_labels(X_opt, U_val, self.Y_val, movement_ind)
                # val_loss = self.loss(Yval_pred.view(-1), Y_val.view(-1)).item()
                # val_losses.append(val_loss)
                val_error = 1 - calc_accuracy_with_movement(Y_val, Yval_pred, movement_ind)
                val_errors.append(val_error)
                if val_error < best_val_error:
                    consecutive_no_improvement = 0
                    best_val_error = val_error

                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 4:
                        print(f"trained {epoch} epochs. val acc: {1 - val_error}")
                        return best_val_error

            if verbose:
                print("------------- epoch %03d / %03d | err: %3.5f" % (epoch + 1, epochs, val_errors[-1]))

    def perceived_test_with_u(self, X, Y, U, delta):
        X_opt, movement_ind = delta.exact(X, True)
        preds = self.f(X_opt)
        Y_tag = self.get_labels(X_opt, U, Y, movement_ind)
        return calc_accuracy_with_movement(Y_tag, preds, movement_ind)

    def collect_perceived_accs(self, data: Data, delta):
        self.results.perceived_acc_record.train_acc = self.perceived_test_with_u(data.X_train, data.Y_train, data.U_train, delta)
        self.results.perceived_acc_record.val_acc = self.perceived_test_with_u(data.X_val, data.Y_val, data.U_val, delta)
        self.results.perceived_acc_record.test_acc = self.perceived_test_with_u(data.X_test, data.Y_test, data.U_test, delta)


class IterativeCausalStrategicTrainer:
    def __init__(self, f_factory: callable, h_factory: callable, h_star: callable, delta: QuadraticCostDelta, dist: Distribution, clean_data: Data, x1_dim: int, x2_dim: int,
                 n_samples_per_round: int, f_loss: callable, h_loss: callable, exploration_params: Optional[ExplorationParams], should_estimate_density: bool, pca: PCA, grid=None,
                 f_reg: float = 0, h_reg: float = 0, cost_scale: float = 1.0, sensitivity_params: SensitivityParams = None, should_test: bool = True, collect_results: bool = True):
        self.f_factory = f_factory
        self.h_factory = h_factory
        self.f = None
        self.f_reg = f_reg
        self.h = None
        self.h_temp = None
        self.h_reg = h_reg
        self.h_star = h_star
        self.delta = delta
        self.cost_scale = cost_scale
        self.clean_data = clean_data
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x_dim = x1_dim + x2_dim
        self.n_samples_per_round = n_samples_per_round
        self.f_loss = f_loss
        self.h_loss = h_loss
        self.pca = pca
        self.ind_dist = InducedDistribution(dist, self.delta, self.x_dim)
        self.h_X_sets = [clean_data.X_train]
        self.h_Y_sets = [clean_data.Y_train]
        self.density_estimator_for_data_collection = estimate_density_torch(clean_data.X_train) if should_estimate_density else None
        self.density_estimator_for_exploration = deepcopy(self.density_estimator_for_data_collection)
        self.test_accs = []
        self.f_list = []
        self.h_list = []
        self.results = []
        self.exploration_params = exploration_params
        self.grid = grid
        self.sensitivity_params = sensitivity_params
        self.should_test = should_test
        self.collect_results = collect_results

    def __learn_h(self, lr, batch_size, epochs, verbose):
        X = torch.vstack(self.h_X_sets)
        Y = torch.vstack(self.h_Y_sets)
        # shuffle the data before split, because without shuffling the samples from last time steps will not be in the training
        X, Y, X_val, Y_val = split_data(X, Y, 0.3, shuffle=True)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        self.h = self.h_factory()
        opt = Adam(self.h.parameters(), lr=lr, weight_decay=self.h_reg)
        trainer = NonStrategicTrainer(self.h, dl, X_val, Y_val, X_val, Y_val, opt, self.h_loss)
        trainer.train(epochs, verbose)
        trainer.test()
        self.calibrate_h(X_val, Y_val)
        self.h_list.append(self.h)

    def calibrate_h(self, X, Y):
        self.h_temp = nn.Parameter(torch.ones(1))
        dl = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.h_temp], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

        logits_list = []
        labels_list = []

        for data in dl:
            inputs, labels = data[0], data[1]

            self.h.eval()
            with torch.no_grad():
                logits_list.append(self.h(inputs))
                labels_list.append(labels)

        # Create tensors
        logits_list = torch.cat(logits_list)
        labels_list = torch.cat(labels_list)
        labels_list = transform_Y_to_zero_one(labels_list)

        def _eval():
            loss = criterion(logits_list * self.h_temp, labels_list)
            loss.backward()
            return loss

        optimizer.step(_eval)

    def __learn_f(self, lr, batch_size, epochs, verbose):
        self.f = self.f_factory()
        dl = DataLoader(TensorDataset(self.clean_data.X_train, self.clean_data.Y_train), batch_size=batch_size, shuffle=True)
        opt = Adam(self.f.parameters(), lr=lr)
        self.delta.cls = self.f
        helper = HelperModel(self.h, self.h_temp)
        trainer = CausalStrategicTrainer(self.f, helper, self.delta, dl, self.clean_data.X_val, self.clean_data.Y_val, None, self.x1_dim, opt, self.f_loss, self.exploration_params,
                                         self.density_estimator_for_exploration, self.f_reg, self.cost_scale)
        trainer.train(epochs, verbose)
        self.f_list.append(self.f)
        if self.collect_results:
            self.results.append(trainer.collect_trainer_results(self.h_star, self.delta, self.clean_data, self.x1_dim))

    def __collect_new_data_from_induced_distribution(self):
        with torch.no_grad():
            delta_inv = QuadraticCostDeltaInverse(self.delta, self.density_estimator_for_data_collection, self.x1_dim)
            X_U = self.ind_dist.sample(self.n_samples_per_round)
            X, U = split_vector(X_U, self.x_dim)
            x1_dim = self.sensitivity_params.true_x1_dim if self.sensitivity_params else self.x1_dim
            X1, _ = split_vector(X, x1_dim)
            with HiddenPrints():
                Y = get_preds(self.h_star(torch.concat([X1, U], 1)))
            X = delta_inv(X[:, self.sensitivity_params.perm]) if self.sensitivity_params else delta_inv(X)
            self.h_X_sets.append(X)
            self.h_Y_sets.append(Y)

    def __update_density_estimator_for_exploration(self):
        weights = None
        if len(self.h_X_sets) > 2:
            clean_samples_weight = len(self.h_X_sets) - 1
            weights_list = [torch.ones(len(self.h_X_sets[0])) * clean_samples_weight] + [torch.ones(len(X_set)) for X_set in self.h_X_sets[1:]]
            weights = torch.cat(tuple(weights_list))
        X = torch.vstack(self.h_X_sets)
        self.density_estimator_for_exploration = estimate_density_torch(X, weights)

    def train(self, time_steps: int, epochs: int, h_lr: float, f_lr: float, batch_size: int, verbose: bool = False, plot_2d: bool = False):
        for t in range(time_steps):
            if verbose:
                print(f"time step: {t+1}\n-----learning h")
            self.__learn_h(h_lr, batch_size, epochs, False)
            if verbose:
                print("\n-----learning f")
            self.__learn_f(f_lr, batch_size, epochs, False)
            if self.should_test:
                acc = test_strategic(self.f, self.h_star, self.delta, self.clean_data.X_test, self.clean_data.U_test, self.x1_dim, False, self.pca,
                                     dens_est=self.density_estimator_for_data_collection)
                self.test_accs.append(acc)
            if t < time_steps - 1:
                if verbose:
                    print("\n-----collecting samples")
                self.__collect_new_data_from_induced_distribution()
                if self.exploration_params and self.exploration_params.should_use:
                    self.exploration_params.coef *= self.exploration_params.decay
                    self.__update_density_estimator_for_exploration()

    def get_best_acc(self):
        return max(self.test_accs)


class IterativeNonStrategicTrainer:
    def __init__(self, f_factory: callable, h_star: callable, delta: QuadraticCostDelta, dist: Distribution, clean_data: Data, x_dim: int, x1_dim: int, n_samples_per_round: int,
                 f_loss: callable, pca: PCA=None, use_only_x1: bool = True):
        self.f_factory = f_factory
        self.f = None
        self.h_star = h_star
        self.delta = delta
        self.clean_data = clean_data
        self.x1_dim = x1_dim
        self.x_dim = x_dim
        self.use_only_x1 = use_only_x1
        self.n_samples_per_round = n_samples_per_round
        self.f_loss = f_loss
        self.pca = pca
        self.ind_dist = InducedDistribution(dist, self.delta, x_dim)
        self.X_sets = [self.clean_data.X_train]
        self.Y_sets = [self.clean_data.Y_train]
        self.w_list = [1]
        self.test_accs = []
        self.results = []

    def __learn_f(self, lr, batch_size, epochs, verbose, use_only_last_samples: bool = False):
        X = self.X_sets[-1] if use_only_last_samples else torch.vstack(self.X_sets)
        Y = self.Y_sets[-1] if use_only_last_samples else torch.vstack(self.Y_sets)
        # shuffle the data before split, because without shuffling the samples from last time steps will not be in the training
        X, Y, X_val, Y_val = split_data(X, Y, 0.3, shuffle=True)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        self.f = self.f_factory()
        self.delta.cls = self.f
        opt = Adam(self.f.parameters(), lr=lr)
        trainer = NonStrategicTrainer(self.f, dl, X_val, Y_val, None, None, opt, self.f_loss)
        trainer.train(epochs)
        self.results.append(trainer.collect_trainer_results(self.h_star, self.delta, self.clean_data, self.x1_dim))

    def __collect_new_data_from_induced_distribution(self):
        with torch.no_grad():
            X_U = self.ind_dist.sample_only_x1(self.n_samples_per_round) if self.use_only_x1 else self.ind_dist.sample(self.n_samples_per_round)
            X, U = split_vector(X_U, self.x1_dim) if self.use_only_x1 else split_vector(X_U, self.x_dim)
            X1 = X if self.use_only_x1 else split_vector(X, self.x1_dim)[0]
            Y = get_preds(self.h_star(torch.concat([X1, U], 1)))
            self.X_sets.append(X)
            self.Y_sets.append(Y)
            n = len(self.X_sets)
            self.w_list = [0.5] + [0.5 / (n - 1) for _ in range(n-1)]

    def train(self, time_steps: int, epochs: int, lr: float, batch_size: int, test_each_step: bool, verbose: bool = False, use_only_last_samples: bool = False):
        for t in range(time_steps):
            if verbose:
                print(f"time step: {t+1}\n-----learning f")
            self.__learn_f(lr, batch_size, epochs, verbose, use_only_last_samples)
            if test_each_step:
                acc = test_strategic(self.f, self.h_star, self.delta, self.clean_data.X_test, self.clean_data.U_test, self.x1_dim, verbose=verbose)
                self.test_accs.append(acc)
            self.__collect_new_data_from_induced_distribution()

    def get_best_acc(self):
        return max(self.test_accs)


class IterativeStrategicTrainer:
    def __init__(self, f_factory: callable, h_star: callable, delta: QuadraticCostDelta, dist: Distribution, clean_data: Data, x_dim: int, x1_dim: int, n_samples_per_round: int,
                 f_loss: callable, cost_scale: float):
        self.f_factory = f_factory
        self.f = None
        self.h_star = h_star
        self.delta = delta
        self.clean_data = clean_data
        self.x1_dim = x1_dim
        self.x_dim = x_dim
        self.cost_scale = cost_scale
        self.n_samples_per_round = n_samples_per_round
        self.f_loss = f_loss
        self.ind_dist = InducedDistribution(dist, self.delta, x_dim)
        self.X_sets = [self.clean_data.X_train]
        self.Y_sets = [self.clean_data.Y_train]
        self.test_accs = []
        self.results = []

    def __learn_f(self, lr, batch_size, epochs, use_only_last_samples: bool = False):
        X = self.X_sets[-1] if use_only_last_samples else torch.vstack(self.X_sets)
        Y = self.Y_sets[-1] if use_only_last_samples else torch.vstack(self.Y_sets)
        # shuffle the data before split, because without shuffling the samples from last time steps will not be in the training
        X, Y, X_val, Y_val = split_data(X, Y, 0.3, shuffle=True)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        self.f = self.f_factory()
        self.delta.cls = self.f
        opt = Adam(self.f.parameters(), lr=lr)
        trainer = StrategicTrainer(self.f, self.delta, dl, X_val, Y_val, None, None, opt, self.f_loss, cost_scale=self.cost_scale)
        trainer.train(epochs, strategic=True)
        self.results.append(trainer.collect_trainer_results(self.h_star, self.delta, self.clean_data, self.x1_dim))

    def __collect_new_data_from_induced_distribution(self):
        with torch.no_grad():
            X_U = self.ind_dist.sample(self.n_samples_per_round)
            X, U = split_vector(X_U, self.x_dim)
            X1 = split_vector(X, self.x1_dim)[0]
            Y = get_preds(self.h_star(torch.concat([X1, U], 1)))
            self.X_sets.append(X)
            self.Y_sets.append(Y)

    def train(self, time_steps: int, epochs: int, lr: float, batch_size: int, test_each_step: bool, verbose: bool = False, use_only_last_samples: bool = False):
        for t in range(time_steps):
            if verbose:
                print(f"time step: {t + 1}\n-----learning f")
            self.__learn_f(lr, batch_size, epochs, use_only_last_samples)
            if test_each_step:
                acc = test_strategic(self.f, self.h_star, self.delta, self.clean_data.X_test, self.clean_data.U_test, self.x1_dim, verbose=verbose)
                self.test_accs.append(acc)
            self.__collect_new_data_from_induced_distribution()

    def get_best_acc(self):
        return max(self.test_accs)
