import argparse
import pickle

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from data import load_card_fraud_data
from delta import QuadraticCostDelta, create_cost_functions
from distribution import RealPositiveAndGenerativeNegetiveDistribution, SequentialDistribution
from model import cross_entropy_loss_for_h, hinge_loss, MLP, StochasticModel
from records import results_list_to_multiple_rounds_stats
from training import NonStrategicTrainer, IterativeCausalStrategicTrainer
from utils import set_seed, split_data, split_vector, create_Y, ExplorationParams, Data


def run(reg_coef, seed, path):
    results_file = f"card_fraud_explore_exp_coef={reg_coef}_seed={seed}"
    print(results_file, flush=True)
    total_dim = 29
    x1_features = [1, 6, 17, 0, 13, 15]
    x2_features = [5, 7, 9, 10, 11, 14]
    u_features = x2_features + list(set([i for i in range(total_dim)]) - set(x1_features) - set(x2_features))
    features_order = x1_features + u_features
    x1_dim = len(x1_features)
    x2_dim = len(x2_features)
    u_dim = 16


    def load_data(seed):
        X, Y = load_card_fraud_data(seed, features_order)
        X = X[:, :x1_dim + u_dim]
        return X, Y

    set_seed(0)
    A = torch.rand(x2_dim, x2_dim)

    def u_to_x2(U):
        # keep only x2 features
        X2 = U[:, :x2_dim]
        # linear transform
        batch_matrix = A.expand(X2.shape[0], X2.shape[1], X2.shape[1])
        batch_vectors = torch.unsqueeze((X2), 2)
        X2 = torch.bmm(batch_matrix, batch_vectors)
        X2 = torch.squeeze(X2, 2)
        # add noise
        return X2

    # create h*
    # load data
    set_seed(0)
    X, Y = load_data(0)
    X_pos, Y_pos = X[Y == 1], Y[Y == 1]
    X_neg, Y_neg = X[Y == -1], Y[Y == -1]
    # sub-sample positives
    n_pos_samples = 500
    X_pos, Y_pos = X_pos[:n_pos_samples], Y_pos[:n_pos_samples]
    X_neg_train, Y_neg_train, X_neg_test, Y_neg_test = split_data(X_neg, Y_neg, 0.2)
    # create balance test
    pos_test_ratio = Y_neg_test.shape[0] / Y_pos.shape[0]
    # split positive samples to train and test
    X_pos_train, Y_pos_train, X_pos_test, Y_pos_test = split_data(X_pos, Y_pos, pos_test_ratio)
    # create test
    X_test, Y_test = torch.concat([X_pos_test, X_neg_test], 0), torch.concat([Y_pos_test, Y_neg_test], 0)
    # create train
    X_train, Y_train = torch.concat([X_pos_train, X_neg_train], 0), torch.concat([Y_pos_train, Y_neg_train], 0)
    # shuffle train
    perm = torch.randperm(X_train.size()[0])
    X_train, Y_train = X_train[perm], Y_train[perm]

    X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train, 0.4)

    # learning params
    batch_size = 64
    lr = 0.01
    epochs = 100
    tau = 4

    h_star_train_ds = TensorDataset(X_train, Y_train)
    h_star_train_dl = DataLoader(h_star_train_ds, batch_size=batch_size, shuffle=True)

    h_star = MLP(in_dim=x1_dim + u_dim, out_dim=1, hidden_dim=10, hidden_layers=3)
    opt = Adam(h_star.parameters(), lr=lr)
    trainer = NonStrategicTrainer(h_star, h_star_train_dl, X_val, Y_val, X_test, Y_test, opt, cross_entropy_loss_for_h)
    trainer.train(epochs)
    print("\nh_star creation:")
    trainer.test()

    s_h_star = StochasticModel(h_star, 0.5, 0.04)

    # samples params
    time_steps = 10
    n_clean_train_samples = 1000
    n_samples_per_round = 200
    n_val_samples = 500
    n_test_samples = 2000

    n_features = x1_dim + x2_dim
    total_n_train_samples = n_clean_train_samples + time_steps * n_samples_per_round
    total_n_val_test_samples = n_val_samples + n_test_samples
    total_n_samples = total_n_train_samples + total_n_val_test_samples
    val_test_frac = total_n_val_test_samples / total_n_samples
    test_frac = n_test_samples / total_n_val_test_samples

    # create distribution
    dist = RealPositiveAndGenerativeNegetiveDistribution(x1_dim, u_to_x2, load_data, seed=0, h_star=s_h_star)
    # create data
    X_U = dist.sample(total_n_samples)
    X, U = split_vector(X_U, n_features)
    Y = create_Y(X, U, s_h_star, x1_dim)
    # split data to train, val, test
    X_U_train, Y_train, X_U_val, Y_val = split_data(X_U, Y, val_test_frac)
    X_U_val, Y_val, X_U_test, Y_test = split_data(X_U_val, Y_val, test_frac)
    X_train, U_train = split_vector(X_U_train, n_features)
    X_val, U_val = split_vector(X_U_val, n_features)
    X_test, U_test = split_vector(X_U_test, n_features)

    # create datasets objects
    clean_data = Data(X_train[:n_clean_train_samples], Y_train[:n_clean_train_samples], U_train[:n_clean_train_samples], X_val, Y_val, U_val, X_test, Y_test, U_test)

    def f_fac():
        set_seed(seed)
        return nn.Linear(n_features, 1)

    def h_fac():
        set_seed(seed)
        return MLP(in_dim=n_features, out_dim=1, hidden_dim=10, hidden_layers=3)

    cost_scale = 1
    cost, _, _ = create_cost_functions(cost_scale, x1_dim, x2_dim)

    print("ICSERM")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    exploration_params = ExplorationParams(should_use=True, coef=reg_coef, decay=0.4)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
    cserm_exp_trainer = IterativeCausalStrategicTrainer(f_fac, h_fac, s_h_star, delta, s_dist, clean_data, x1_dim, x2_dim, n_samples_per_round, hinge_loss, cross_entropy_loss_for_h,
                                                        exploration_params=exploration_params, should_estimate_density=True, pca=None, cost_scale=cost_scale)
    cserm_exp_trainer.train(time_steps, epochs, lr, lr, batch_size, verbose=True)
    stats = results_list_to_multiple_rounds_stats(cserm_exp_trainer.results)

    # save results
    stats.to_csv(f"{path}/{results_file}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('reg_coef')
    parser.add_argument('seed')
    parser.add_argument('path')
    args = parser.parse_args()
    print(f"seed: {args.seed}, reg_coef: {args.reg_coef}")
    reg_coef = float(args.reg_coef)
    seed = int(args.seed)
    path = args.path
    run(reg_coef, seed, path)

