import numpy as np
import random
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *

rpy.verbosity(0)

# seed = 42
# np.random.seed(seed)
# reservoir_seeds = [42,24] # seed reservoirs?

# Hyperparams

runs = 1 # number of runs
coupling_num = 11 # number of coupling strengths to test
train_sample = 0.5 # training data size; range of interest: 0.5 (non-overlapping) -- 1 (completely overlapping)
r1_size = 500 # reservoir 1 size
r2_size = r1_size # reservoir 2 size


# Curate training data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# discard signals shorter than 10 timesteps
X_train = [signal for signal in X_train if len(signal) >= 10]
Y_train = [signal for signal in Y_train if len(signal) >= 10]
X_test = [signal for signal in X_test if len(signal) >= 10]
Y_test = [signal for signal in Y_test if len(signal) >= 10]

# sample portion of training data
sampled_indices1 = np.random.choice(len(X_train), size=int(len(X_train) * train_sample), replace=False)
X_train1 = [X_train[i] for i in sampled_indices1]
Y_train1 = [Y_train[i] for i in sampled_indices1]

if train_sample==0.5:
    sampled_indices2 = np.setdiff1d(np.arange(len(X_train)), sampled_indices1) # sample (non-overlapping)
    np.random.shuffle(sampled_indices2)
    X_train2 = [X_train[i] for i in sampled_indices2]
    Y_train2 = [Y_train[i] for i in sampled_indices2]
else:
    sampled_indices2 = np.random.choice(len(X_train), size=int(len(X_train) * train_sample), replace=False) # sample (random)
    X_train2 = [X_train[i] for i in sampled_indices2]
    Y_train2 = [Y_train[i] for i in sampled_indices2]


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
        model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=c)

        model.train_r1(input=X_train1, target=Y_train1, warmup=5, reset='random')
        model.train_r2(input=X_train2, target=Y_train2, warmup=5, reset='random')

        results = model.test(input=X_test, target=Y_test, reset='random')

        # calculate accuracy
        y1_final = [signal[-1] for signal in results[0]]
        y2_final = [signal[-1] for signal in results[1]]
        targets_final = [signal[-1] for signal in Y_test]

        acc1 = [np.argmax(test) == np.argmax(pred1) for (test, pred1) in zip(targets_final, y1_final)]
        acc2 = [np.argmax(test) == np.argmax(pred2) for (test, pred2) in zip(targets_final, y2_final)]
        acc_joint = [np.argmax(test) == np.argmax(np.mean((pred1, pred2), axis=0)) for (test, pred1, pred2) in zip(targets_final, y1_final, y2_final)]

        joint, y1, y2 = np.mean(acc_joint), np.mean(acc1), np.mean(acc2)

        acc_joint.append(joint)
        acc_y1.append(y1)
        acc_y2.append(y2)

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
plt.ylabel('Avg. Pcorrect')
plt.grid(True)
plt.legend(loc='upper right')
# plt.savefig(f'plt_figs/performance.png', dpi=300)
plt.show()

