import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *

rpy.verbosity(0)

# seed = 42
# np.random.seed(seed)
# reservoir_seeds = [42,24] # seed reservoirs?

# Hyperparams

runs = 10 # number of runs
coupling_num = 11 # number of coupling strengths to test
train_sample = 1.0 # training data size; range of interest: 0.5 (non-overlapping) -- 1 (completely overlapping)
r1_size = 500 # reservoir 1 size
r2_size = r1_size # reservoir 2 size


# Curate training data

X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = japanese_vowels(repeat_targets=True)

# # standardize LPCs
# scaler = StandardScaler().fit(np.vstack(X_train_raw)) # fit only to training data
# 
# X_train_scaled = [scaler.transform(signal) for signal in X_train_raw]
# X_test_scaled = [scaler.transform(signal) for signal in X_test_raw]

# predict next-timestep LPC coefficients
X_train = [signal[:-1] for signal in X_train_raw]
Y_train = [signal[1:] for signal in X_train_raw]
X_test = [signal[:-1] for signal in X_test_raw]
Y_test = [signal[1:] for signal in X_test_raw]

# # discard signals shorter than 10 timesteps
# X_train = [signal for signal in X_train if len(signal) >= 10]
# Y_train = [signal for signal in Y_train if len(signal) >= 10]
# X_test = [signal for signal in X_test if len(signal) >= 10]
# Y_test = [signal for signal in Y_test if len(signal) >= 10]

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

        model.train_r1(input=X_train1, target=Y_train1, warmup=2, reset='zero')
        model.train_r2(input=X_train2, target=Y_train2, warmup=2, reset='zero')

        results = model.test(input=X_test, target=Y_test, reset='zero')

        # # convert values back to raw LPCs
        # y1 = [scaler.inverse_transform(signal) for signal in results[0]]
        # y2 = [scaler.inverse_transform(signal) for signal in results[1]]
        # yjoint = [scaler.inverse_transform(np.mean((signal1, signal2), axis=0)) for (signal1, signal2) in zip(y1, y2)]
        # targets = [scaler.inverse_transform(signal) for signal in Y_test]
        
        # use base values
        y1 = results[0]
        y2 = results[1]
        yjoint = [np.mean((signal1, signal2), axis=0) for (signal1, signal2) in zip(y1, y2)]
        targets = Y_test

        # calculate avg rmse per signal
        rmse_1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(targets, y1)]
        rmse_2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(targets, y2)]
        rmse_joint = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(targets, yjoint)]

        # # calculate avg rmse per LPC per signal
        # rmse_1 = [np.sqrt(np.mean((test - pred)**2, axis=0)) for (test, pred) in zip(targets, y1)]
        # rmse_2 = [np.sqrt(np.mean((test - pred)**2, axis=0)) for (test, pred) in zip(targets, y2)]
        # rmse_joint = [np.sqrt(np.mean((test - pred)**2, axis=0)) for (test, pred) in zip(targets, yjoint)]

        # avg over signals
        acc_joint.append(np.mean(rmse_joint, axis=0))
        acc_y1.append(np.mean(rmse_1, axis=0))
        acc_y2.append(np.mean(rmse_2, axis=0))

    store_couplings.append(c)
    store_accuracies_joint.append(acc_joint)
    store_accuracies_y1.append(acc_y1)
    store_accuracies_y2.append(acc_y2)
    
    print(f'N: {runs}; CS: {c}; RMSE: {np.mean(acc_joint)}')


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


# Plot per LPC channel

# x_values = np.array(store_couplings)
# 
# datasets = [
#     np.array(store_accuracies_joint), 
#     np.array(store_accuracies_y1), 
#     np.array(store_accuracies_y2)
# ]
# labels = [r'$ESN_{joint}$', r'$ESN_1$', r'$ESN_2$']
# colors = ['black', 'black', 'black']
# linestyles = ['-', '--', ':']
# 
# fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
# axes = axes.flatten()   # makes it easy to index subplots
# 
# n_lpcs = 12
# 
# for lpc_idx in range(n_lpcs):
#     ax = axes[lpc_idx]
# 
#     for y_values, label, color, ls in zip(datasets, labels, colors, linestyles):
#         # y_values has shape (n_couplings, n_runs, 12)
#         # We want just this LPC dimension
#         y_lpc = y_values[:, :, lpc_idx]
# 
#         # mean across runs
#         y_means = np.mean(y_lpc, axis=1)
# 
#         # SEM across runs
#         y_sems = np.std(y_lpc, axis=1, ddof=1) / np.sqrt(y_lpc.shape[1])
# 
#         ax.plot(x_values, y_means, linestyle=ls, color=color, label=label)
#         ax.fill_between(x_values, y_means - y_sems, y_means + y_sems,
#                         color='gray', alpha=0.3)
# 
#     ax.set_title(f"LPC {lpc_idx+1}")
#     ax.grid(True)
# 
# for ax in axes[8:]:   # bottom row
#     ax.set_xlabel("Coupling Strength")
# for ax in axes[::4]:  # first column
#     ax.set_ylabel("Avg. RMSE")
# 
# fig.legend(labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
# 
# plt.tight_layout()
# plt.show()

