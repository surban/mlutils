from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from mlutils import gather


def imshow_grid(data, ny=None, nx=None, cmap='gray_r', interpolation='none', aspect=1.3,
                x_spacing=1, y_spacing=1):
    """
    Plots each sample from data in a grid.
    :param data: 3-dimensional data array to plot. 3rd dimension is the sample axis.
    :type data: np.ndarray
    """
    x_spacing = int(x_spacing)
    y_spacing = int(y_spacing)
    space_value = np.max(data)

    if data.ndim != 3:
        raise ValueError("data must be 3 dimensional")
    h, w, n = data.shape[0], data.shape[1], data.shape[2]

    if ny is None:
        ny = round(sqrt(w * n / (aspect * float(h))))
    ny = int(ny)
    if ny == 0:
        ny = 1

    if nx is None:
        nx = round(n / float(ny))
    nx = int(nx)

    while nx * ny < n:
        if nx < ny:
            nx += 1
        else:
            ny += 1
    assert nx * ny >= n

    y_stride = h + y_spacing
    x_stride = w + x_spacing
    grid = np.full((ny * y_stride - y_spacing, nx * x_stride - x_spacing), space_value)
    for iy in range(ny):
        for ix in range(nx):
            i = iy * nx + ix
            if i < n:
                grid[iy * y_stride : iy * y_stride + h, ix * x_stride : ix * x_stride + w] = data[:, :, i]

#    plt.imshow(grid, cmap=cmap, interpolation=interpolation,
#               extent=(-0.5, nx - 0.5, ny - 0.5, -0.5), aspect=(nx * x_stride) / float(ny * y_stride))
    plt.imshow(grid, cmap=cmap, interpolation=interpolation,
               extent=(-0.5, nx - 0.5, ny - 0.5, -0.5))
    plt.gca().set_aspect('auto')

    def ticks(l):
        if l == 1:
            return [0]
        elif l <= 10:
            step = 1
        elif l <= 100:
            step = 5
        else:
            step = 10
        return np.append(np.arange(0, l, step), [l - 1])

    plt.xticks(ticks(nx))
    plt.yticks(ticks(ny))


def plot_weight_histograms(ps, weights_to_plot, bins=10):
    """
    Plots histograms of the given weights.
    :param ps: ParameterSet
    :param weights_to_plot: list of weights to plot
    """
    for i, par in enumerate(weights_to_plot):
        plt.subplot(len(weights_to_plot), 1, i)
        plt.hist(gather(ps[par]).flat, bins=bins)
        plt.xlabel(par)
    plt.tight_layout()


def plot_binary_sequence(seq, true_seq=None, gray=None):
    """
    Plots a binary sequence and optionally compares it with another sequence.
    :param seq: seq[ch, step] - sequence to plot
    :param true_seq: true_seq[ch, step] - sequence to compare with
    :param gray: gray[step] - background is colored gray where gray > 0
    """
    n_channels = seq.shape[0]
    n_steps = seq.shape[1]

    if true_seq is not None:
        # calculate note diff
        diff = np.ones((n_channels, n_steps, 3))
        for step in range(n_steps):
            for ch in range(n_channels):
                if seq[ch, step] == true_seq[ch, step]:
                    if seq[ch, step] == 1:
                        diff[ch, step, :] = 0           # black means matchs with note
                    elif seq[ch, step] == 0:
                        if gray is not None and gray[step] > 0:
                            diff[ch, step, :] = 0.7
                        else:
                            diff[ch, step, :] = 1           # white means matchs with no note
                    else:
                        assert False
                elif seq[ch, step] == 1:
                    diff[ch, step, 0:2] = 0         # blue means new note
                elif seq[ch, step] == 0:
                    diff[ch, step, 1:] = 0          # red means note deleted
                else:
                    assert False

        plt.imshow(diff, origin='lower', aspect='auto', interpolation='nearest',
                   extent=(0, n_steps, 0, n_channels))
    else:
        plt.imshow(seq, origin='lower', aspect='auto', interpolation='nearest', cmap=plt.cm.gray_r,
                   extent=(0, n_steps, 0, n_channels))

    plt.xlabel('step')
    plt.ylabel('channel')