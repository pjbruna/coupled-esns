import numpy as np
import random
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *

rpy.verbosity(0)

seed = 42
np.random.seed(seed)

# Hyperparams

reservoir_seeds = [42,42] # seed reservoirs?
train_sample = 1.0 # training data size; range of interest: 0.5 (non-overlapping) -- 1 (completely overlapping)


# Curate training data

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


# Run model

model = CesnModel(r1_nnodes=500, r2_nnodes=500, coupling_strength=0, is_seed=reservoir_seeds)

model.train_r1(input=X_train1, target=Y_train1, warmup=0, reset='zero')
model.train_r2(input=X_train2, target=Y_train2, warmup=0, reset='zero')

results = model.test(input=X_train, target=Y_train, noise_scale=0, reset='zero')

joint, y1, y2 = model.accuracy(pred1=results[0], pred2=results[1], target=Y_train)

print(f'ESN_joint: {joint}')
print(f'ESN_1:{y1}')
print(f'ESN_2:{y2}')

