import math
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn

from utils import split_vector, get_preds

COLORS = dict(LIGHT_GREEN="#bcedbb", DARK_GREEN="#00ff7f", LIGHT_RED="#e3acb6", DARK_RED="#dc143c", LIGHT_BLUE="#8fb2eb", BLUE="blue", GHOSTS_GREEN="#46703e")


def plot_dataset_2d(X, y, pca):
    if X.shape[1] > 2:
        X = pca.transform(X)
    elif len(X[0]) == 1:
        X = torch.concat([X, torch.zeros_like(X)], 1)
    fig = plt.figure(1, figsize=(4, 3))
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
    plt.show()


def plot_with_lda(X, Y):
    lda = LinearDiscriminantAnalysis()
    X_2d = lda.fit_transform(X, Y)
    fig = plt.figure(1, figsize=(4, 3))
    plt.scatter(X_2d, Y, marker="o", c=Y, s=25, edgecolor="k")
    plt.show()


def plot_2d_with_movement(f, h, h_star, delta, X, U, x1_dim, ax=None, image_scale=5, title=None, cost_scale=None, title_pad=-30, h_label_location=None, f_label_location=None):
    X_opt, movement_ind = delta.exact(X, True)
    X1_opt, _ = split_vector(X_opt, x1_dim)
    X_ghosts = X[movement_ind.view(-1) == 1]
    X1_ghosts, _ = split_vector(X_ghosts, x1_dim)
    U_ghosts = U[movement_ind.view(-1) == 1]
    with torch.no_grad():
        new_Y = get_preds(h_star(torch.concat([X1_opt, U], 1)))
        Y_ghosts = get_preds(h_star(torch.concat([X1_ghosts, U_ghosts], 1)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax = plot_syth_exp(ax, X_opt.detach().clone(), new_Y, X_ghosts=X_ghosts, Y_ghosts=Y_ghosts, h_star=h_star, f=f, h=h, title=title, image_scale=image_scale, movement_line=True,
                       cost_scale=cost_scale, title_pad=title_pad, h_label_location=h_label_location, f_label_location=f_label_location)
    return ax


class MovementLine:
    def __init__(self, f: nn.Linear, cost_scale):
        self.f = f
        self.max_move = math.sqrt(2 / cost_scale)
        self.W = f.weight[0]

    def __call__(self, X):
        return self.f(X) + self.max_move * torch.norm(self.W, p=2)


def plot_syth_exp(ax, X, y, X_ghosts=None, Y_ghosts=None, h_star=None, f=None, h=None, title=None, additional_func: Tuple[callable, str] = None, image_scale=5, movement_line=False,
                  cost_scale=None, title_pad=-30, h_label_location=None, f_label_location=None):
    x1_l, x1_h = -image_scale, image_scale
    x2_l, x2_h = -image_scale, image_scale
    # create grid
    x1_range = torch.linspace(x1_l, x1_h, 100)
    x2_range = torch.linspace(x2_l, x2_h, 100)
    x1_grid, x2_grid = torch.meshgrid(x1_range, x2_range)
    # Flatten the grid coordinates into a single vector
    xy_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)
    if h_star:
        def hard_h_star(x):
            y = h_star(x)
            y[y < 0] = -1
            y[y > 0] = 1
            return y
        h_star_contours = create_contours_for_model(ax, hard_h_star, xy_grid, x1_grid, x2_grid, [COLORS["LIGHT_RED"], COLORS["LIGHT_GREEN"]],
                                                    levels=[-1, 0, 1], fill=True, kwargs={"zorder": 0})
        # h_star_contours = create_contours_for_model(ax, hard_h_star, xy_grid, x1_grid, x2_grid, ["blue"], ["h*"], levels=[0])
    if f:
        f_contours = create_contours_for_model(ax, f, xy_grid, x1_grid, x2_grid, ["red"], [r"$f$"], kwargs={"zorder": 2}, label_location=f_label_location)
        # add movement line
        if movement_line:
            create_contours_for_model(ax, MovementLine(f, cost_scale), xy_grid, x1_grid, x2_grid, ["purple"], kwargs={'alpha': 0.5, 'linestyles': 'dashed'})

    if h:
        h_contours = create_contours_for_model(ax, h, xy_grid, x1_grid, x2_grid, [COLORS["BLUE"]], [r"$h$"], kwargs={"zorder": 2}, label_location=h_label_location)
    if additional_func:
        create_contours_for_model(ax, additional_func[0], xy_grid, x1_grid, x2_grid, ["black"], [additional_func[1]])

    colors = [COLORS["DARK_RED"] if y_i == -1 else COLORS["DARK_GREEN"] for y_i in y]
    ax.scatter(X[:, 0], X[:, 1], marker="o", c=colors, s=25, edgecolor="k", zorder=3)
    if X_ghosts is not None:
        ghosts_colors = [COLORS["DARK_RED"] if y_i == -1 else COLORS["GHOSTS_GREEN"] for y_i in Y_ghosts]
        ax.scatter(X_ghosts[:, 0], X_ghosts[:, 1], s=25, facecolors='none', edgecolors=ghosts_colors, zorder=1, alpha=0.4)
    ax.set_xlim(x1_l, x1_h)
    ax.set_ylim(x2_l, x2_h)
    # ax.set_xlabel(r"$x_c$", fontsize=20)
    # ax.set_ylabel(r"$x_r$", fontsize=20)
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    if title:
        ax.set_title(title, y=1.0, pad=title_pad, fontdict={"size": 18} )
    return ax


def plot_2d_grid(test_accs, f_list, h_list, delta, X, U, h_star, time_steps, image_scale=5, cost_scale=None):
    fig, axs = plt.subplots(nrows=math.ceil(time_steps / 4), ncols=4, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    axs = axs.ravel()
    for time_step in range(time_steps):
        ax = axs[time_step]
        acc = test_accs[time_step]
        f = f_list[time_step]
        h = h_list[time_step]
        delta.cls = f
        # X_opt, movement_ind = delta.exact(X, True)
        # X1_opt, _ = split_vector(X_opt, 1)
        # with torch.no_grad():
        #     new_Y = get_preds(h_star(torch.concat([X1_opt, U], 1)))
        title = f"Time: {time_step + 1}, Acc: {round(acc, 2)}"
        # ax = plot_syth_exp(ax, X_opt.detach().clone(), new_Y, h_star=h_star, f=f, h=h, title=title)
        plot_2d_with_movement(f, h, h_star, delta, X, U, 1, ax, image_scale, title, cost_scale)
    for ax in axs[time_steps:]:
        fig.delaxes(ax)
    plt.show()


def create_contours_for_model(ax, model, grid_vector, x1_grid, x2_grid, colors, labels=None, levels=None, fill=False, kwargs={}, label_location=None):
    if levels is None:
        levels = [0]
    # Compute output for each point in the grid
    scores = model(grid_vector)
    # Reshape the model output into a grid
    scores_grid = scores.reshape(x1_grid.shape)
    # Plot the decision boundary
    if fill:
        contours = ax.contourf(x1_grid.detach().numpy(), x2_grid.detach().numpy(), scores_grid.detach().numpy(), levels=levels, colors=colors, **kwargs)
    else:
        contours = ax.contour(x1_grid.detach().numpy(), x2_grid.detach().numpy(), scores_grid.detach().numpy(), levels=levels, colors=colors, **kwargs)
    # set label
    if labels:
        fmt = {contours.levels[i]: labels[i] for i in range(len(levels))}
        ax.clabel(contours, fmt=fmt, fontsize=15, inline_spacing=1,  manual=label_location if label_location is not None else False)
    return contours


def plot_models_sequence(models, model_name, h_star):
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(4, 4))
    x1_l, x1_h = -5, 5
    x2_l, x2_h = -5, 5
    # create grid
    x1_range = torch.linspace(x1_l, x1_h, 100)
    x2_range = torch.linspace(x2_l, x2_h, 100)
    x1_grid, x2_grid = torch.meshgrid(x1_range, x2_range)
    # Flatten the grid coordinates into a single vector
    xy_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)
    # colors
    cmap = plt.cm.get_cmap('BuPu')
    start = 40
    jumps = 15
    colors = [cmap(i) for i in range(start, jumps * len(models) + start, jumps)]
    for i in range(len(models)):
        scores = models[i](xy_grid)
        # Reshape the model output into a grid
        scores_grid = scores.reshape(x1_grid.shape)
        # Plot the decision boundary
        contours = ax.contour(x1_grid.detach().numpy(), x2_grid.detach().numpy(), scores_grid.detach().numpy(), levels=[0], colors=[colors[i]])
        contours.collections[0].set_label(r"$h_{" + f"{i+1}" + r"}$")

    scores = h_star(xy_grid)
    # Reshape the model output into a grid
    scores_grid = scores.reshape(x1_grid.shape)
    # Plot the decision boundary
    contours = ax.contour(x1_grid.detach().numpy(), x2_grid.detach().numpy(), scores_grid.detach().numpy(), levels=[0], colors=["black"])
    contours.collections[0].set_label(r"$h^*$")

    ax.set_xlim(x1_l, x1_h)
    ax.set_ylim(x2_l, x2_h)
    ax.set_xlabel(r"$x_c$", fontsize=20)
    ax.set_ylabel(r"$x_r$", fontsize=20)
    ax.set_title(f"${model_name}$ across time", fontsize=17)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=13)
    plt.show()


def plot_coefs(coefs, maxv=1, normalize=True, bias=None):
    n_coefs = len(coefs)
    x = [f"{i}" for i in range(n_coefs)]
    if bias:
        coefs = torch.concat([coefs, bias])
        x += ["b"]
    coefs_to_plot = F.normalize(coefs, dim=0) if normalize else coefs
    fig = plt.figure(figsize=(10, 4))
    plt.bar(x, coefs_to_plot, color='maroon',
            width=0.4)
    plt.xticks(x)
    ax = plt.gca()
    if maxv:
        ax.set_ylim([-maxv, maxv])
    plt.show()
