import numpy as np
import random
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import santafe_laser, to_forecasting
from cesn_model import *

rpy.verbosity(0)


# Hyperparams

runs = 5 # number of runs
coupling_num = 11 # number of coupling strengths to test
reservoir_seeds = None # [42,24] # seed reservoirs?

r1_size = 500 # reservoir 1 size
r2_size = r1_size # reservoir 2 size


# Curate training data

data = santafe_laser()

lag = 1 # forecasting time lag
sample = int(len(data)/2) # proportion for training

X_train = [data[:sample]]
Y_train = [data[lag:sample+lag]]
X_test = [data[:len(data)-lag]]
Y_test = [data[lag:]]


# Run model

store_couplings = []
store_accuracies_joint = []
store_accuracies_y1 = []
store_accuracies_y2 = []

for c in np.linspace(0.0, 1.0, num=coupling_num):
    c = round(c, 1)
    acc_joint = []
    acc_y1 = []
    acc_y2 = []

    for r in range(runs):
        print(f'Simulation #{r}')
        model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=c, is_seed=reservoir_seeds)

        model.train_r1(input=X_train, target=Y_train)
        model.train_r2(input=X_train, target=Y_train)

        results = model.test(input=X_test, target=Y_test, noise_scale=0, do_print=False, save_reservoir=False)

        rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, results[0])]
        rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, results[1])]

        acc_joint.append(np.mean((rmse1, rmse2), axis=0)[0])
        acc_y1.append(rmse1[0])
        acc_y2.append(rmse2[0])

    store_couplings.append(c)
    store_accuracies_joint.append(acc_joint)
    store_accuracies_y1.append(acc_y1)
    store_accuracies_y2.append(acc_y2)
    
    print(f'N: {runs}; CS: {c}; R1: {r1_size}; R2: {r2_size};  Acc: {np.mean(acc_joint)}')


# Plot

x_values = np.array(store_couplings)
datasets = [np.array(store_accuracies_joint), np.array(store_accuracies_y1), np.array(store_accuracies_y2)]
labels = [r'$ESN_{joint}$', r'$ESN_1$', r'$ESN_2$']

colors = ['black', 'black', 'black']
linestyles = ['-', '--', ':']

for y_values, label, color, ls in zip(datasets, labels, colors, linestyles):
    if y_values.shape[1] == 1:
        y_means = y_values.flatten()
        plt.plot(x_values, y_means, linestyle=ls, color=color, label=label)
    else:
        y_means = np.mean(y_values, axis=1)
        y_sems = np.std(y_values, axis=1, ddof=1) / np.sqrt(y_values.shape[1])
        plt.plot(x_values, y_means, linestyle=ls, color=color, label=label)
        plt.fill_between(x_values, y_means - y_sems, y_means + y_sems,
                         color='gray', alpha=0.3)

        
plt.xlabel('Coupling Strength')
plt.ylabel('Avg. RMSE')
plt.grid(True)
plt.legend(loc='upper right')
# plt.savefig(f'plt_figs/performance.png', dpi=300)
plt.show()

