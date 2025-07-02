import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def plot_coupling_strengths(x=None, y=None, do_print=False, save=False):
    x_values = np.array(x)
    y_values = np.array(y)

    if y_values.shape[1] == 1:
        # Only one observation per point — no error bars
        y_means = y_values.flatten()
        plt.plot(x_values, y_means, 'o-', color='b')
    else:
        y_means = np.mean(y_values, axis=1)
        y_sems = np.std(y_values, axis=1, ddof=1) / np.sqrt(y_values.shape[1])
        plt.errorbar(x_values, y_means, yerr=y_sems, fmt='o-', color='b', ecolor='gray', capsize=5)

    plt.xlabel('Coupling Strength')
    plt.ylabel('Avg. Pcorrect')
    # plt.title()
    # plt.legend()
    plt.grid(True)

    if do_print==True:
        plt.show()

    if save!=False:
        plt.savefig(save)


def plot_coupling_with_comparison(x=None, y_joint=None, y1=None, y2=None, do_print=False, save=False):
    x_values = np.array(x)
    datasets = [np.array(y_joint), np.array(y1), np.array(y2)]
    labels = ['esn_joint', 'esn_1', 'esn_2']

    viridis = get_cmap('viridis')
    colors = [viridis(i) for i in np.linspace(0, 1, len(datasets))]

    for y_values, label, color in zip(datasets, labels, colors):
        if y_values.shape[1] == 1:
            # Only one observation per point — no error bars
            y_means = y_values.flatten()
            plt.plot(x_values, y_means, 'o-', color=color, label=label)
        else:
            y_means = np.mean(y_values, axis=1)
            y_sems = np.std(y_values, axis=1, ddof=1) / np.sqrt(y_values.shape[1])
            plt.errorbar(x_values, y_means, yerr=y_sems, fmt='o-', color=color,
                         ecolor='gray', capsize=5, label=label)


    plt.xlabel('Coupling Strength')
    plt.ylabel('Avg. Pcorrect')
    plt.grid(True)
    plt.legend()

    if do_print:
        plt.show()

    if save:
        plt.savefig(save)