from copy import deepcopy

from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from delta import QuadraticCostDelta, create_cost_functions
from distribution import SequentialDistribution
from model import cross_entropy_loss_for_h, hinge_loss
from testing import test_strategic
from training import NonStrategicTrainer, StrategicTrainer, IterativeCausalStrategicTrainer
from utils import ExplorationParams, set_seed, local_seed, split_data, split_vector, create_Y, create_uniform_samples, Data


class Experiment2d:
    def __init__(self,
                 f_lr: float,
                 h_lr: float,
                 batch_size: int,
                 epochs: int,
                 cost_scale: float,
                 tau: float,
                 time_steps: int,
                 n_clean_train_samples: int,
                 n_samples_per_round: int,
                 total_n_val_test_samples: int,
                 exploration_params: ExplorationParams,
                 dist,
                 u_to_x2: callable,
                 h_creator: callable,
                 h_star: callable,
                 f_reg: float,
                 h_reg: float,
                 seed: int
                 ):
        self.f_lr = f_lr
        self.h_lr = h_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.cost_scale = cost_scale
        self.tau = tau
        self.x1_dim = 1
        self.x2_dim = 1
        self.n_features = 2
        self.time_steps = time_steps
        self.n_clean_train_samples = n_clean_train_samples
        self.n_samples_per_round = n_samples_per_round
        self.exploration_params = exploration_params
        self.dist = dist
        self.h_creator = h_creator
        self.h_star = h_star
        self.f_reg = f_reg
        self.h_reg = h_reg
        self.seed = seed
        self.cost, _, _ = create_cost_functions(cost_scale, self.x1_dim, self.x2_dim)

        def f_fac():
            set_seed(seed)
            return nn.Linear(self.n_features, 1)

        def h_fac():
            set_seed(seed)
            return self.h_creator()

        self.f_fac = f_fac
        self.h_fac = h_fac

        total_n_train_samples = n_clean_train_samples + time_steps * n_samples_per_round
        total_n_samples = total_n_train_samples + total_n_val_test_samples
        val_test_frac = total_n_val_test_samples / total_n_samples
        test_frac = 0.8

        with local_seed(0):
            # create h_star over X1,X2
            X_U_grid, Y_grid = create_uniform_samples(-6, 6, 300, self.n_features, self.x1_dim, u_to_x2, h_star)
            X_grid, _ = split_vector(X_U_grid, self.n_features)
            X_grid, Y_grid, X_tmp_val, Y_tmp_val = split_data(X_grid, Y_grid, 0.2)
            train_ds_tmp = TensorDataset(X_grid, Y_grid)
            train_dl_tmp = DataLoader(train_ds_tmp, batch_size=batch_size, shuffle=True)
            h_star_x1x2 = h_creator()
            opt = Adam(h_star_x1x2.parameters(), lr=h_lr, weight_decay=h_reg)
            trainer = NonStrategicTrainer(h_star_x1x2, train_dl_tmp, X_tmp_val, Y_tmp_val, None, None, opt, cross_entropy_loss_for_h)
            trainer.train(epochs)
        self.h_star_x1x2 = h_star_x1x2

        with local_seed(0):
            X_U = dist.sample(total_n_samples)
            X, U = split_vector(X_U, self.n_features)
            Y = create_Y(X, U, h_star, self.x1_dim)
            X_U, Y, X_Uval, Yval = split_data(X_U, Y, val_test_frac)
            X_Uval, Yval, X_Utest, Ytest = split_data(X_Uval, Yval, test_frac)
            self.X_U = X_U
            self.X, self.U = split_vector(X_U, self.n_features)
            self.Y = Y
            self.Xval, self.Uval = split_vector(X_Uval, self.n_features)
            self.Yval = Yval
            self.Xtest, self.Utest = split_vector(X_Utest, self.n_features)
            self.Ytest = Ytest

            self.full_train_dl = DataLoader(TensorDataset(self.X, self.Y), batch_size=batch_size, shuffle=True)
            self.full_X_U_train_dl = DataLoader(TensorDataset(self.X_U, self.Y), batch_size=batch_size, shuffle=True)
            # the iterative methods use clean_data. we give them only n_samples_per_round from it because they sample additional data in the future
            self.clean_data = Data(self.X[:n_clean_train_samples], self.Y[:n_clean_train_samples], self.U[:n_clean_train_samples], self.Xval, self.Yval, self.Uval, self.Xtest, self.Ytest, self.Utest)

    def delta(self, f):
        return QuadraticCostDelta(f, self.cost, self.tau)

    def run_erm(self):
        f = self.f_fac()
        opt = Adam(f.parameters(), lr=self.f_lr)
        delta = QuadraticCostDelta(f, self.cost, self.tau)
        trainer = NonStrategicTrainer(f, self.full_train_dl, self.Xval, self.Yval, self.Xtest, self.Ytest, opt, hinge_loss)
        trainer.train(self.epochs, False)
        acc = trainer.test()
        print(f"Non-strategic test: {acc} accuracy")
        acc = test_strategic(f, self.h_star, delta, self.Xtest, self.Utest, self.x1_dim, verbose=False)
        print(f"Strategic test: {acc} accuracy")
        return f

    def run_serm(self):
        f = self.f_fac()
        opt = Adam(f.parameters(), lr=self.f_lr)
        delta = QuadraticCostDelta(f, self.cost, self.tau)
        trainer = StrategicTrainer(f, delta, self.full_train_dl, self.Xval, self.Yval, self.Xtest, self.Ytest, opt, hinge_loss, self.f_reg, self.cost_scale)
        trainer.train(self.epochs, True)
        acc = test_strategic(f, self.h_star, delta, self.Xtest, self.Utest, self.x1_dim,  verbose=False)
        print(f"Strategic test: {acc} accuracy")
        return f

    def run_cserm(self):
        set_seed(self.seed)
        delta = QuadraticCostDelta(None, self.cost, self.tau)
        # create dist object from the samples that not included in the partial clean data
        s_dist = SequentialDistribution(self.x1_dim, self.X_U[self.n_clean_train_samples:])
        trainer = IterativeCausalStrategicTrainer(self.f_fac, self.h_fac, self.h_star, delta, s_dist, self.clean_data, self.x1_dim, self.x2_dim, self.n_samples_per_round,
                                                  hinge_loss, cross_entropy_loss_for_h, deepcopy(self.exploration_params),
                                                  True, None, f_reg=self.f_reg, h_reg=self.h_reg, cost_scale=self.cost_scale)
        trainer.train(self.time_steps, self.epochs, self.h_lr, self.f_lr, self.batch_size, True, plot_2d=True)
        return trainer.f_list, trainer.h_list, trainer.test_accs


