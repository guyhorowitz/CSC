import argparse

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from card_fraud_params import params
from data import load_card_fraud_data
from delta import QuadraticCostDelta, create_cost_functions
from distribution import RealPositiveAndGenerativeNegetiveDistribution, SequentialDistribution
from model import cross_entropy_loss_for_h, hinge_loss, MLP, HelperModel, StochasticModel
from records import get_chosen_model_and_multiple_rounds_stats
from training import NonStrategicTrainer, StrategicTrainer, IterativeCausalStrategicTrainer, IterativeNonStrategicTrainer, IterativeStrategicTrainer, CausalStrategicTrainerWithU
from utils import set_seed, split_data, split_vector, create_Y, ExplorationParams, Data


def run(cost_scale, seed, path):
    results_file = f"cost_scale={cost_scale}_seed={seed}"
    print(results_file, flush=True)

    # learning params
    batch_size = params["batch_size"]
    lr = params["lr"]
    epochs = params["epochs"]
    tau = params["tau"]
    small_lamda = params["small_lamda"]
    large_lamda = params["large_lamda"]
    lamda_decay = params["lamda_decay"]

    # samples params
    time_steps = params["time_steps"]
    n_clean_train_samples = params["n_clean_train_samples"]
    n_samples_per_round = params["n_samples_per_round"]
    n_val_samples = params["n_val_samples"]
    n_test_samples = params["n_test_samples"]

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
    h_star_train_ds = TensorDataset(X_train, Y_train)
    h_star_train_dl = DataLoader(h_star_train_ds, batch_size=batch_size, shuffle=True)

    h_star = MLP(in_dim=x1_dim + u_dim, out_dim=1, hidden_dim=10, hidden_layers=3)
    opt = Adam(h_star.parameters(), lr=lr)
    trainer = NonStrategicTrainer(h_star, h_star_train_dl, X_val, Y_val, X_test, Y_test, opt, cross_entropy_loss_for_h)
    trainer.train(100)
    print("\nh_star creation:")
    trainer.test()

    s_h_star = StochasticModel(h_star, 0.5, 0.04)

    # create simulation data
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
    # random shuffle the data to obtain random splits
    set_seed(seed)
    X_U = X_U[torch.randperm(X_U.size()[0])]

    X, U = split_vector(X_U, n_features)
    Y = create_Y(X, U, s_h_star, x1_dim)
    # split data to train, val, test
    X_U_train, Y_train, X_U_val, Y_val = split_data(X_U, Y, val_test_frac)
    X_U_val, Y_val, X_U_test, Y_test = split_data(X_U_val, Y_val, test_frac)
    X_train, U_train = split_vector(X_U_train, n_features)
    X_val, U_val = split_vector(X_U_val, n_features)
    X_test, U_test = split_vector(X_U_test, n_features)
    # split x1, x2
    X1_train, X2_train = split_vector(X_train, x1_dim)
    X1_val, X2_val = split_vector(X_val, x1_dim)
    X1_test, X2_test = split_vector(X_test, x1_dim)

    # create datasets objects
    full_data = Data(X_train, Y_train, U_train, X_val, Y_val, U_val, X_test, Y_test, U_test)
    clean_data = Data(X_train[:n_clean_train_samples], Y_train[:n_clean_train_samples], U_train[:n_clean_train_samples], X_val, Y_val, U_val, X_test, Y_test, U_test)
    X1_clean_data = Data(X1_train[:n_clean_train_samples], Y_train[:n_clean_train_samples], U_train[:n_clean_train_samples], X1_val, Y_val, U_val, X1_test, Y_test, U_test)
    X2_full_data = Data(X2_train, Y_train, U_train, X2_val, Y_val, U_val, X2_test, Y_test, U_test)

    full_train_dl = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    X_U_train_dl = DataLoader(TensorDataset(X_U_train, Y_train), batch_size=batch_size, shuffle=True)

    # only x1, x2
    full_X1_train_dl = DataLoader(TensorDataset(X1_train, Y_train), batch_size=batch_size, shuffle=True)
    full_X2_train_dl = DataLoader(TensorDataset(X2_train, Y_train), batch_size=batch_size, shuffle=True)

    def f_fac():
        set_seed(seed)
        return nn.Linear(n_features, 1)

    def x1_f_fac():
        set_seed(seed)
        return nn.Linear(x1_dim, 1)

    def x2_f_fac():
        set_seed(seed)
        return nn.Linear(x2_dim, 1)

    def h_fac():
        set_seed(seed)
        return MLP(in_dim=n_features, out_dim=1, hidden_dim=10, hidden_layers=3)

    cost, x1_cost, x2_cost = create_cost_functions(cost_scale, x1_dim, x2_dim)

    # erm
    print("ERM")
    f = f_fac()
    opt = Adam(f.parameters(), lr=lr)
    erm_trainer = NonStrategicTrainer(f, full_train_dl, X_val, Y_val, X_test, Y_test, opt, hinge_loss)
    erm_trainer.train(epochs, False)
    delta = QuadraticCostDelta(f, cost, tau)
    erm_chosen_model_stats = erm_trainer.collect_trainer_results(s_h_star, delta, full_data, x1_dim).calc_stats("ERM")

    # rrm - all
    print("RRM - all")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U[n_clean_train_samples:])
    rrm_all_trainer = IterativeNonStrategicTrainer(f_fac, s_h_star, delta, s_dist, clean_data, n_features, x1_dim, n_samples_per_round, hinge_loss, use_only_x1=False)
    rrm_all_trainer.train(time_steps, epochs, lr, batch_size, test_each_step=True, verbose=True, use_only_last_samples=False)
    rrm_all_chosen_model_stats, rrm_all_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(rrm_all_trainer.results, "ERM_all")

    # rrm - last
    print("RRM - last")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U[n_clean_train_samples:])
    rrm_last_trainer = IterativeNonStrategicTrainer(f_fac, s_h_star, delta, s_dist, clean_data, n_features, x1_dim, n_samples_per_round, hinge_loss, use_only_x1=False)
    rrm_last_trainer.train(time_steps, epochs, lr, batch_size, test_each_step=True, verbose=True, use_only_last_samples=True)
    rrm_last_chosen_model_stats, rrm_last_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(rrm_last_trainer.results, "ERM_last")

    # ERMc
    print("ERMc")
    set_seed(seed)
    delta = QuadraticCostDelta(None, x1_cost, tau)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U[n_clean_train_samples:])
    erm_c_trainer = IterativeNonStrategicTrainer(x1_f_fac, s_h_star, delta, s_dist, X1_clean_data, n_features, x1_dim, n_samples_per_round, hinge_loss, use_only_x1=True)
    erm_c_trainer.train(time_steps, epochs, lr, batch_size, test_each_step=True, verbose=True, use_only_last_samples=False)
    erm_c_chosen_model_stats, erm_c_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(erm_c_trainer.results, "ERMc")

    # SERMr
    print("SERMr:")
    f = x2_f_fac()
    opt = Adam(f.parameters(), lr=lr)
    delta = QuadraticCostDelta(f, x2_cost, tau)
    serm_r_trainer = StrategicTrainer(f, delta, full_X2_train_dl, X2_val, Y_val, X2_test, Y_test, opt, hinge_loss, cost_scale=cost_scale)
    serm_r_trainer.train(epochs, True)
    serm_r_res = serm_r_trainer.collect_trainer_results_using_only_x2(s_h_star, delta, full_data, x1_dim)
    serm_r_chosen_model_stats = serm_r_res.calc_stats("SERMr")

    # SERM
    print("SERM:")
    f = f_fac()
    opt = Adam(f.parameters(), lr=lr)
    delta = QuadraticCostDelta(f, cost, tau)
    serm_trainer = StrategicTrainer(f, delta, full_train_dl, X_val, Y_val, X_test, Y_test, opt, hinge_loss, cost_scale=cost_scale)
    serm_trainer.train(epochs, True)
    serm_chosen_model_stats = serm_trainer.collect_trainer_results(s_h_star, delta, full_data, x1_dim).calc_stats("SERM")

    # RSRM - all
    print("RSRM - all")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
    rsrm_all_trainer = IterativeStrategicTrainer(f_fac, s_h_star, delta, s_dist, clean_data, n_features, x1_dim, n_samples_per_round, hinge_loss, cost_scale)
    rsrm_all_trainer.train(time_steps, epochs, lr, batch_size, test_each_step=True, verbose=True, use_only_last_samples=False)
    rsrm_all_chosen_model_stats, rsrm_all_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(rsrm_all_trainer.results, "RSRM_all")

    # RSRM - last
    print("RSRM - last")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
    rsrm_last_trainer = IterativeStrategicTrainer(f_fac, s_h_star, delta, s_dist, clean_data, n_features, x1_dim, n_samples_per_round, hinge_loss, cost_scale)
    rsrm_last_trainer.train(time_steps, epochs, lr, batch_size, test_each_step=True, verbose=True, use_only_last_samples=True)
    rsrm_last_chosen_model_stats, rsrm_last_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(rsrm_last_trainer.results, "RSRM_last")

    # CSERM
    print("CSERM")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
    cserm_trainer = IterativeCausalStrategicTrainer(f_fac, h_fac, s_h_star, delta, s_dist, clean_data, x1_dim, x2_dim, n_samples_per_round, hinge_loss, cross_entropy_loss_for_h,
                                                    exploration_params=None, should_estimate_density=True, pca=None, cost_scale=cost_scale)
    cserm_trainer.train(time_steps, epochs, lr, lr, batch_size, verbose=True)
    cserm_chosen_model_stats, cserm_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(cserm_trainer.results, "CSERM")

    # CSERM exp small
    print("ICSERM - exp small")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    exploration_params = ExplorationParams(should_use=True, coef=small_lamda, decay=lamda_decay)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
    cserm_exp_small_trainer = IterativeCausalStrategicTrainer(f_fac, h_fac, s_h_star, delta, s_dist, clean_data, x1_dim, x2_dim, n_samples_per_round, hinge_loss,
                                                              cross_entropy_loss_for_h,
                                                              exploration_params=exploration_params, should_estimate_density=True, pca=None, cost_scale=cost_scale)
    cserm_exp_small_trainer.train(time_steps, epochs, lr, lr, batch_size, verbose=True)
    cserm_exp_small_chosen_model_stats, cserm_exp_small_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(cserm_exp_small_trainer.results, "CSERM_exp_small")

    # CSERM exp large
    print("ICSERM - exp large")
    set_seed(seed)
    delta = QuadraticCostDelta(None, cost, tau)
    exploration_params = ExplorationParams(should_use=True, coef=large_lamda, decay=lamda_decay)
    # create dist object from the samples that not included in the partial clean data
    s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
    cserm_exp_large_trainer = IterativeCausalStrategicTrainer(f_fac, h_fac, s_h_star, delta, s_dist, clean_data, x1_dim, x2_dim, n_samples_per_round, hinge_loss,
                                                              cross_entropy_loss_for_h,
                                                              exploration_params=exploration_params, should_estimate_density=True, pca=None, cost_scale=cost_scale)
    cserm_exp_large_trainer.train(time_steps, epochs, lr, lr, batch_size, verbose=True)
    cserm_exp_large_chosen_model_stats, cserm_exp_large_multi_rounds_stats = get_chosen_model_and_multiple_rounds_stats(cserm_exp_large_trainer.results, "CSERM_exp_large")

    # CSERM - h*
    print("CSERM - h*")
    f = f_fac()
    opt = Adam(f.parameters(), lr=lr)
    delta = QuadraticCostDelta(f, cost, tau)
    helper = HelperModel(h_star)
    cserm_h_star_trainer = CausalStrategicTrainerWithU(f, helper, delta, X_U_train_dl, full_train_dl, X_val, X_U_val, Y_val, X_test, n_features, x1_dim, opt, hinge_loss,
                                                       cost_scale=cost_scale)
    cserm_h_star_trainer.train(epochs, False)
    cserm_h_star_chosen_model_stats = cserm_h_star_trainer.collect_trainer_results(s_h_star, delta, full_data, x1_dim).calc_stats("CSERM_h_star")

    chosen_models_stats_list = [erm_chosen_model_stats, rrm_all_chosen_model_stats, rrm_last_chosen_model_stats, erm_c_chosen_model_stats, serm_r_chosen_model_stats,
                                serm_chosen_model_stats, rsrm_all_chosen_model_stats, rsrm_last_chosen_model_stats, cserm_chosen_model_stats,
                                cserm_exp_small_chosen_model_stats, cserm_exp_large_chosen_model_stats, cserm_h_star_chosen_model_stats]

    multi_rounds_stats_list = [rrm_all_multi_rounds_stats, rrm_last_multi_rounds_stats, erm_c_multi_rounds_stats, rsrm_all_multi_rounds_stats, rsrm_last_multi_rounds_stats,
                               cserm_multi_rounds_stats, cserm_exp_small_multi_rounds_stats, cserm_exp_large_multi_rounds_stats]

    chosen_models_df = pd.concat(chosen_models_stats_list)
    multi_rounds_df = pd.concat(multi_rounds_stats_list)
    # save results
    chosen_models_df.to_csv(f"{path}/chosen_models_{results_file}.csv")
    multi_rounds_df.to_csv(f"{path}/multi_rounds_results_{results_file}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cost_scale')
    parser.add_argument('seed')
    parser.add_argument('path')
    args = parser.parse_args()
    print(f"seed: {args.seed}, cost_scale: {args.cost_scale}")
    cost_scale = float(args.cost_scale)
    seed = int(args.seed)
    path = args.path
    run(cost_scale, seed, path)
