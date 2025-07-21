import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *

rpy.verbosity(0)


def run_model(runs=None, coupling=None, inequality=None, r1_size=None, r2_size=None, train_sample=None):

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

    gain = []

    for r in range(runs):
        print(f'Simulation #{r}')

        model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=coupling)

        model.train_r1(input=X_train1, target=Y_train1)
        model.train_r2(input=X_train2, target=Y_train2)

        results = model.test_unequal(input=X_test, target=Y_test, noise_scale_1=0, noise_scale_2=inequality, do_print=False, save_reservoir=False)
        
        joint, y1, y2 = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test, do_print=False)
        
        gain.append(joint - max(y1, y2))

    return np.mean(gain)



###################################



# Load dataset

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# Run model

run_num = 10

inequality_strength = []
coupling_strength = []
measure = []

for noise in np.arange(0.0, 1.01, 0.1): 
    for cs in np.arange(0.0, 1.01, 0.1):
        noise = round(noise, 1)
        cs = round(cs, 1)

        print(f'Running: inequality={noise}, coupling={cs}')
        result = run_model(runs=run_num, coupling=cs, inequality=noise, r1_size=500, r2_size=500, train_sample=0.5)

        inequality_strength.append(noise)
        coupling_strength.append(cs)
        measure.append(result)

# Store data

df = pd.DataFrame({
    'inequality_strength': inequality_strength,
    'coupling_strength': coupling_strength,
    'measure': measure
})

# Save to CSV
df.to_csv(f'data/gainsyn_fitness_landscape_N={run_num}.csv', index=False)
