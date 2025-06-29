import numpy as np
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *
from functions import *

rpy.verbosity(0)


# Define model

def run_model(runs=None, coupling_num=None, r1_size=None, r2_size=None):
    store_couplings = []
    store_accuracies = []

    for c in np.linspace(0.0, 1.0, num=coupling_num):
        acc = []

        for r in range(runs):
            model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=int(c))

            model.train_r1(input=X_train1, target=Y_train1)
            model.train_r2(input=X_train2, target=Y_train2)

            results = model.test(input=X_test, target=Y_test, noise=0, do_print=False, save_reservoir=False)

            acc.append(model.accuracy(pred1=results[0], pred2=results[1], target=Y_test, do_print=False))

        store_couplings.append(c)
        store_accuracies.append(acc)

        # print(f'N: {runs}; CS: {c}; R1: {r1_size}; R2: {r2_size};  Acc: {np.mean(acc)}')

    return store_couplings, store_accuracies


# Curate training data

train_sample = 0.5 # training data size; range of interest: 0.5 (non-overlapping) -- 1 (completely overlapping)

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

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


# Run



for nnodes in np.linspace(100, 1000, num=10):
    run_model(runs=25, coupling_num=11, r1_size=nnodes, r2_size=nnodes)