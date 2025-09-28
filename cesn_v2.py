import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *
 

# seed = 42
# np.random.seed(seed)
# reservoir_seeds = [seed,seed]

save_data = False
rpy.verbosity(0)


# Hyperparams

runs = 25
train_sample = 1 # proportion of training data; range: 0.5 (non-overlapping) -- 1 (completely overlapping)
noise = [0, 0] # noise added to input signals during testing (equal vs unequal conditions)
rsize = 500
reset_state = 'zero' # reset reservoirs between signals


# Curate training data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

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


# # identical (vs complementary) training:
# X_train2 = X_train1
# Y_train2 = Y_train1


# Run model

store_couplings = []
store_accuracies_joint = []
store_accuracies_y1 = []
store_accuracies_y2 = []

for c in ["independent", "interaction"]:
    acc_joint = []
    acc_y1 = []
    acc_y2 = []

    for r in range(runs):
        print(f'Simulation #{r}')
        model = CesnModel_V2(r1_nnodes=rsize, r2_nnodes=rsize)

        model.train_r1(input=X_train1, target=Y_train1, reset=reset_state)
        model.train_r2(input=X_train2, target=Y_train2, reset=reset_state)

        results = model.test(input=X_test, target=Y_test, condition=c, input_sigma=noise, reset=reset_state)

        joint, y1, y2 = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test)

        acc_joint.append(joint)
        acc_y1.append(y1)
        acc_y2.append(y2)

        # Save

        if save_data==True:
            signal_ids = [i for i, signal in enumerate(Y_test) for _ in range(len(signal))]

            sig_df = pd.DataFrame(signal_ids, columns=['signal'])
            sig_df.to_csv(f'data/v2/signal_ids.csv', index=False)

            ytest = np.vstack(Y_test)
            ytest_df = pd.DataFrame(ytest, columns=[f'X{i}' for i in range(len(ytest[0]))])
            ytest_df.to_csv(f'data/v2/ytest.csv', index=False)

            y1 = np.vstack(results[0])
            y2 = np.vstack(results[1])
            yjoint = np.vstack([np.mean((pred1, pred2), axis=0) for (pred1, pred2) in zip(results[0], results[1])])
            # r1 = np.vstack(results[2])
            # r2 = np.vstack(results[3])

            for (name, item) in zip(['y1', 'y2', 'yjoint'], [y1, y2, yjoint]): # r1, r2
                df = pd.DataFrame(item, columns=[f'X{i}' for i in range(len(item[0]))])
                df.to_csv(f'data/v2/{name}_{c}_r={rsize}_train={train_sample}_reset={reset_state}_noise={noise}_sim={r}.csv', index=False)

    store_couplings.append(c)
    store_accuracies_joint.append(acc_joint)
    store_accuracies_y1.append(acc_y1)
    store_accuracies_y2.append(acc_y2)
    
    print(f'{c}; nnodes: {rsize};  N: {runs}; pcorrect: {np.mean(acc_joint)}')


# Plot

import numpy as np
import matplotlib.pyplot as plt

# Define categorical x-axis
x_categories = ['independent', 'interaction']
x_indices = np.arange(len(x_categories))  # [0, 1]

datasets = [np.array(store_accuracies_joint), 
            np.array(store_accuracies_y1), 
            np.array(store_accuracies_y2)]
labels = [r'$ESN_{joint}$', r'$ESN_1$', r'$ESN_2$']

colors = ['black', 'black', 'black']
linestyles = ['-', '--', ':']

for y_values, label, color, ls in zip(datasets, labels, colors, linestyles):
    if y_values.shape[1] == 1:
        y_means = y_values.flatten()
        plt.plot(x_indices, y_means, linestyle=ls, color=color, label=label, marker='o')
    else:
        y_means = np.mean(y_values, axis=1)
        y_sems = np.std(y_values, axis=1, ddof=1) / np.sqrt(y_values.shape[1])
        plt.plot(x_indices, y_means, linestyle=ls, color=color, label=label, marker='o')
        plt.fill_between(x_indices, y_means - y_sems, y_means + y_sems,
                         color='gray', alpha=0.3)

plt.xticks(x_indices, x_categories)  # Set category labels on x-axis
plt.xlabel('Coupling Type')
plt.ylabel('Avg. Pcorrect')
plt.grid(True)
plt.legend(loc='upper right')
# plt.savefig(f'plt_figs/v2/performance_runs={runs}_reset={reset_state}_warmup={wrmup}_train={train_sample}_inequality={ineq}.png', dpi=300)
plt.show()
