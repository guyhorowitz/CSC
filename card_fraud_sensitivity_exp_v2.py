import argparse
import pickle
from random import sample

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from data import load_card_fraud_data
from delta import QuadraticCostDelta, create_cost_functions
from distribution import RealPositiveAndGenerativeNegetiveDistribution, SequentialDistribution
from model import cross_entropy_loss_for_h, hinge_loss, MLP, StochasticModel, HelperModel
from sensitivity_exp_utils import SensitivityParams, generate_permutations_for_feature_type_flips
from testing import test
from training import NonStrategicTrainer, IterativeCausalStrategicTrainer, CausalStrategicTrainer
from utils import set_seed, split_data, split_vector, create_Y, get_preds, Data


def run(n_flips, seed, path):
    results_file = f"card_fraud_sensitivity_exp_v2_n_flips={n_flips}_seed={seed}"
    print(results_file, flush=True)
    # split features
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

    # learning params
    batch_size = 64
    lr = 0.01
    epochs = 100
    tau = 4

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

    # create simualtive data
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

    accs, prec_accs = [], []
    max_permutations = 30
    possible_permutations = generate_permutations_for_feature_type_flips(x1_dim, x2_dim, n_flips)
    set_seed(seed)
    sampled_permutations = sample(possible_permutations, k=max_permutations) if len(possible_permutations) > max_permutations else possible_permutations
    for perm, assumed_x1_dim in sampled_permutations:
        print(f"perm: {perm}, x1_assumed_dim: {assumed_x1_dim}", flush=True)
        assumed_x2_dim = n_features - assumed_x1_dim
        perm = torch.LongTensor(perm)
        perm_inv = torch.argsort(perm)
        s_X_train, s_X_val, s_X_test = X_train[:, perm], X_val[:, perm], X_test[:, perm]
        sensitivity_params = SensitivityParams(perm=perm, perm_inv=perm_inv, true_x1_dim=x1_dim)

        # create datasets objects
        clean_data = Data(s_X_train[:n_clean_train_samples], Y_train[:n_clean_train_samples], U_train[:n_clean_train_samples], s_X_val, Y_val, U_val, X_test, Y_test, U_test)

        def f_fac():
            set_seed(seed)
            return nn.Linear(n_features, 1)

        def h_fac():
            set_seed(seed)
            return MLP(in_dim=n_features, out_dim=1, hidden_dim=10, hidden_layers=3)

        cost_scale = 1
        cost, _, _ = create_cost_functions(cost_scale, x1_dim, x2_dim)

        set_seed(seed)
        delta = QuadraticCostDelta(None, cost, tau)
        # create dist object from the samples that not included in the partial clean data
        s_dist = SequentialDistribution(x1_dim, X_U_train[n_clean_train_samples:])
        cserm_trainer = IterativeCausalStrategicTrainer(f_fac, h_fac, s_h_star, delta, s_dist, clean_data, assumed_x1_dim, assumed_x2_dim, n_samples_per_round, hinge_loss, cross_entropy_loss_for_h,
                                                        exploration_params=None, should_estimate_density=True, pca=None, cost_scale=cost_scale, sensitivity_params=sensitivity_params,
                                                        should_test=False, collect_results=False)
        cserm_trainer.train(time_steps, epochs, lr, lr, batch_size, verbose=True)

        def test_for_sensitivity(model, h_star, delta, s_X_test, U_test):
            with torch.no_grad():
                # after movement
                s_X_opt, movement_ind = delta.exact(s_X_test, True)
                X_opt = s_X_opt[:, perm_inv]
                X1_opt, _ = split_vector(X_opt, x1_dim)
                Y_after_movement = get_preds(h_star(torch.concat([X1_opt, U_test], 1)))

                acc = test(model, s_X_opt, Y_after_movement, movement_ind)
                return acc

        val_accs, test_accs = [], []
        for f in cserm_trainer.f_list:
            delta = QuadraticCostDelta(f, cost, tau)
            val_acc = test_for_sensitivity(f, s_h_star, delta, s_X_val, U_val)
            test_acc = test_for_sensitivity(f, s_h_star, delta, s_X_test, U_test)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

        chosen_model_idx = np.argmax(val_accs)
        acc_of_chosen_model = test_accs[chosen_model_idx]
        # get precieved acc
        f = cserm_trainer.f_list[chosen_model_idx]
        h = HelperModel(cserm_trainer.h_list[chosen_model_idx])
        delta.cls = f
        temp_trainer = CausalStrategicTrainer(f, h, delta, train_dl=None, X_val=None, Y_val=None, X_test=None, x1_dim=assumed_x1_dim, opt=None, loss=None, exploration_params=None,
                                              cost_scale=cost_scale)
        prec_acc = temp_trainer.perceived_test(X_test, Y_test, delta)

        accs.append(acc_of_chosen_model)
        prec_accs.append(prec_acc)

        # save results
    results = {"accs": accs, "prec_accs": prec_accs}
    with open(f"{path}/{results_file}.pkl", "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_flips')
    parser.add_argument('seed')
    parser.add_argument('path')
    args = parser.parse_args()
    n_flips = int(args.n_flips)
    seed = int(args.seed)
    path = args.path
    run(n_flips, seed, path)
