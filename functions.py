import numpy as np
import matplotlib.pyplot as plt

def plot_coupling_strengths(x=None, y=None, title=None, do_print=False, save=False,):
    x_values = np.array(x)
    y_values = np.array(y)

    y_means = np.mean(y_values, axis=1)
    y_sems = np.std(y_values, axis=1, ddof=1) / np.sqrt(y_values.shape[1])

    plt.errorbar(x_values, y_means, yerr=y_sems, fmt='o-', color='b', ecolor='gray', capsize=5)

    plt.xlabel('Coupling Strength')
    plt.ylabel('Avg. Pcorrect')
    plt.title(title)
    # plt.legend()
    plt.grid(True)

    if do_print==True:
        plt.show()

    if save==True:
        plt.savefig('plt_figs/acc_x_coupling.png')
