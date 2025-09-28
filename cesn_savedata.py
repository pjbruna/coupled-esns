import numpy as np
import random
import pandas as pd
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *

rpy.verbosity(0)

# seed = 42
# np.random.seed(seed)


# Hyperparams

run_num = 10 # number of runs
# reservoir_seeds = None # [42, 24] # seed reservoirs?
coupling = 0.0 # coupling strength
train_sample = 0.5 # training data size; range of interest: 0.5 (non-overlapping) -- 1 (completely overlapping)
r1_size = 500 # reservoir 1 size
r2_size = r1_size # reservoir 2 size
# inequality = 0 # unequal noise scale


# Curate training data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# predict next-timestep LPC coefficients
Y_train = [signal[1:] for signal in X_train]
Y_test = [signal[1:] for signal in X_test]

X_train = [signal[:-1] for signal in X_train]
X_test = [signal[:-1] for signal in X_test]

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

for run in range(run_num):
    print(f"Running simulation {run + 1}/10")

    model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=coupling)

    model.train_r1(input=X_train1, target=Y_train1, warmup=0, reset='zero')
    model.train_r2(input=X_train2, target=Y_train2, warmup=0, reset='zero')

    results = model.test(input=X_test, target=Y_test, reset='zero')
    # results = model.test_unequal(input=X_test, target=Y_test, noise_scale_1=0, noise_scale_2=inequality, do_print=False, save_reservoir=False)


    # Save

    signal_ids = [i for i, signal in enumerate(Y_test) for _ in range(len(signal))]
    
    sig_df = pd.DataFrame(signal_ids, columns=['signal'])
    sig_df.to_csv(f'data/prediction/signal_ids.csv', index=False)

    ytest = np.vstack(Y_test)
    ytest_df = pd.DataFrame(ytest, columns=[f'X{i}' for i in range(len(ytest[0]))])
    ytest_df.to_csv(f'data/prediction/ytest.csv', index=False)

    y1 = np.vstack(results[0])
    y2 = np.vstack(results[1])
    yjoint = np.vstack([np.mean((pred1, pred2), axis=0) for (pred1, pred2) in zip(results[0], results[1])])
    # r1 = np.vstack(results[2])
    # r2 = np.vstack(results[3])

    for (name, item) in zip(['y1', 'y2', 'yjoint'], [y1, y2, yjoint]): # r1, r2
        df = pd.DataFrame(item, columns=[f'X{i}' for i in range(len(item[0]))])
        df.to_csv(f'data/prediction/{name}_cs={coupling}_r1={r1_size}_r2={r2_size}_sample={train_sample}_sim={run}.csv', index=False)
