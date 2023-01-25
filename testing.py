import torch
from torch import nn

from plot_data import plot_dataset_2d
from utils import calc_accuracy_with_movement, calc_accuracy, get_preds, HiddenPrints, split_vector


def test(model: nn.Module, X_test, Y_test, movement_ind=None, verbose=True):
    with torch.no_grad():
        Ytest_pred = model(X_test)
        if torch.isnan(Ytest_pred).any():
            raise Exception("preds contains nan")
        acc = calc_accuracy_with_movement(Y_test, Ytest_pred, movement_ind) if movement_ind is not None else calc_accuracy(Y_test, Ytest_pred)
        if verbose:
            print(f"\tAccuracy: {acc}")
        return acc


def test_with_h_star(model, h_star, X, X_1, U, movement_ind, verbose=True):
    with torch.no_grad():
        Y = get_preds(h_star(torch.concat([X_1, U], 1)))
        mean_score_of_moving_points = h_star(torch.concat([X_1, U], 1)[(movement_ind == 1).view(-1)]).mean()
        # print(f"\tmean score of h_star on moving points: {mean_score_of_moving_points}")
        return test(model, X, Y, movement_ind, verbose)


def test_strategic(model, h_star, delta, X_test, U_test, x1_dim, plot=False, pca=None, verbose=True, dens_est=None):
    with torch.no_grad():
        with HiddenPrints():
            # before movement
            X1, _ = split_vector(X_test, x1_dim)
            Y_before_movement = get_preds(h_star(torch.concat([X1, U_test], 1)))
        # after movement
        X_opt, movement_ind = delta.exact(X_test, True)
        X1_opt, _ = split_vector(X_opt, x1_dim)
        Y_after_movement = get_preds(h_star(torch.concat([X1_opt, U_test], 1)))

        total_move = movement_ind.sum().item()
        move_percent = movement_ind.mean().item()
        n_positive_move = movement_ind[Y_before_movement == 1].sum().item()
        n_negative_move = movement_ind[Y_before_movement == -1].sum().item()
        positive_move_percent = n_positive_move / total_move if total_move > 0 else 0
        negative_move_percent = n_negative_move / total_move if total_move > 0 else 0
        n_positive_move_to_negative = (Y_after_movement[Y_before_movement == 1] == -1).sum().item()
        n_negative_move_to_positive = (Y_after_movement[Y_before_movement == -1] == 1).sum().item()
        positive_move_to_negative_percent = n_positive_move_to_negative / n_positive_move if n_positive_move > 0 else 0
        negative_move_to_positive_percent = n_negative_move_to_positive / n_negative_move if n_negative_move > 0 else 0
        if dens_est:
            dens_score = torch.exp(dens_est.predict(X_opt[(movement_ind == 1).view(-1), :])).mean()
        if verbose:
            print(f"\ttotal move: {round(move_percent, 3)}, pos move (out of total move): {round(positive_move_percent, 3)}, "
                  f"neg move (out of total move): {round(negative_move_percent, 3)},\n\tpos turn to neg (out of pos move): {positive_move_to_negative_percent}, "
                  f"neg turn to pos (out of neg move): {negative_move_to_positive_percent},\n\t "
                  f"f.bias: {round(model.bias.data.item(), 3)}", flush=True
                  )
            if dens_est:
                print(f"\tmean likelihood of moving points: {dens_score}")
        acc = test(model, X_opt, Y_after_movement, movement_ind, verbose)
        if plot:
            new_Y = get_preds(h_star(torch.concat([X1_opt, U_test], 1)))
            plot_dataset_2d(X_opt.detach().clone(), new_Y, pca)
        return acc


def test_strategic_with_h(model, h, delta, X_test, x1_dim):
    X_opt, movement_ind = delta.exact(X_test, True)
    X1_opt, _ = split_vector(X_opt, x1_dim)
    _, X2 = split_vector(X_test, x1_dim)
    with torch.no_grad():
        Y = get_preds(h(torch.concat([X1_opt, X2], 1)))
    acc = test(model, X_opt, Y, movement_ind, verbose=False)
    print(f"\tAccuracy with h and x_r: {acc}")
    return acc


def test_strategic_only_X2_model(model, h_star, delta, X1_test, X2_test, U_test, plot=False, pca=None):
    X_opt, movement_ind = delta.exact(X2_test, True)
    acc = test_with_h_star(model, h_star, X_opt, X1_test, U_test, movement_ind)
    if plot:
        with torch.no_grad():
            new_Y = get_preds(h_star(torch.concat([X1_test, U_test], 1)))
        plot_dataset_2d(X_opt.detach().clone(), new_Y, pca)
    return acc