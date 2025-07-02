import numpy as np
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *
from functions import *

rpy.verbosity(0)


def run_model(runs=None, coupling_num=None, noise=None, r1_size=None, r2_size=None, train_sample=None):

    # Training data

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

    # Model

    # store_couplings = []
    # store_accuracies = []

    avg_acc = []

    for c in [0.0, 0.5]: # np.linspace(0.0, 1.0, num=coupling_num):
        acc = []

        for r in range(runs):
            print(f'Simulation #{r}')

            model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=c)

            model.train_r1(input=X_train1, target=Y_train1)
            model.train_r2(input=X_train2, target=Y_train2)

            results = model.test(input=X_test, target=Y_test, noise_scale=noise, do_print=False, save_reservoir=False)

            joint = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test, do_print=False)[0]

            acc.append(joint)

        # store_couplings.append(c)
        # store_accuracies.append(acc)

        avg_acc.append(np.mean(acc))

        # print(f'N: {runs}; CS: {c}; R1: {r1_size}; R2: {r2_size};  Acc: {np.mean(acc)}')

    # half = np.argmin(np.abs(np.linspace(0.0, 1.0, num=coupling_num) - 0.5))
    syn_gain_diff = avg_acc[1] - avg_acc[0]

    # syn_gain_area = np.trapz(avg_acc - avg_acc[0], dx=coupling_num)

    return syn_gain_diff



###################################



# Load dataset

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# Run model

run_num = 10
noise_num = 0

res_size = []
train_size = []
measure = []

for nnode in [250, 500, 750]: 
    for sample in [0.5, 0.75, 1.0]:
        print(f'Running: nnode={nnode}, sample={sample}')
        result = run_model(runs=run_num, noise=noise_num, r1_size=nnode, r2_size=nnode, train_sample=sample)

        res_size.append(nnode)
        train_size.append(sample)
        measure.append(result)

# Plot heat map

res_size_unique = sorted(set(res_size))
train_size_unique = sorted(set(train_size))

# Create a 2D grid for the heatmap
heatmap = np.full((len(train_size_unique), len(res_size_unique)), np.nan)

# Fill the grid with measure values
for r, t, m in zip(res_size, train_size, measure):
    i = train_size_unique.index(t)
    j = res_size_unique.index(r)
    heatmap[i, j] = m

# Plot the heatmap
plt.figure(figsize=(8, 6))
im = plt.imshow(heatmap, cmap='viridis', origin='lower', aspect='auto')

# Set axis labels and ticks
plt.xticks(ticks=np.arange(len(res_size_unique)), labels=res_size_unique)
plt.yticks(ticks=np.arange(len(train_size_unique)), labels=train_size_unique)
plt.xlabel('Reservoir Size (nodes)')
plt.ylabel('Training Size (%)')
# plt.title(f'N={run_num}')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Gain')

plt.tight_layout()
plt.savefig(f'plt_figs/diff_heatmap_N={run_num}_noise={noise_num}.png')


