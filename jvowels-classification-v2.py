import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import reservoirpy as rpy
import multiprocessing as mp
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat
from reservoirpy.observables import rmse, rsquare
from reservoirpy.datasets import japanese_vowels
from functions import *

rpy.verbosity(0)


## DEFINE MODEL ##


def run_models(index):
    print(f'Running model: {index}')

    # seed = 42
    # rpy.set_seed(seed)
    # print(f'Seed: {seed}')


    ## TRAIN ESNs ##


    reservoir1 = Reservoir(r1_nnodes, sr=0.9, lr=0.1, activation='tanh')
    readout1 = Ridge(ridge=1e-6)

    reservoir2 = Reservoir(r2_nnodes, sr=0.9, lr=0.1, activation='tanh')
    readout2 = Ridge(ridge=1e-6)

    print(f'Run: {index}, R1 Name: {reservoir1.name}')
    print(f'Run: {index}, R2 Name: {reservoir2.name}')

    train_states1 = []
    train_states2 = []

    for (x, y) in zip(X_train, Y_train):
        r1 = np.array([np.zeros(reservoir1.output_dim)] * len(x))
        r2 = np.array([np.zeros(reservoir2.output_dim)] * len(x))

        for t in range(len(x)):
            if t==0:
                input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
            else:
                input_fb = np.concatenate((x[t], fbs * y[t]), axis=None)

            r1[t] = reservoir1.run(input_fb)
            r2[t] = reservoir2.run(input_fb)

        # log training states
        train_states1.append(r1)
        train_states2.append(r2)

        # reset reservoirs
        reservoir1.reset(np.random.random(reservoir1.state().shape)) # reservoir1.reset()
        reservoir2.reset(np.random.random(reservoir2.state().shape)) # reservoir2.reset()

    readout1 = readout1.fit(train_states1, Y_train)
    readout2 = readout2.fit(train_states2, Y_train)


    ## TEST ##


    mask1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1]) # np.array([1,1,1,1,1,1,0,0,0,0,0,0])
    mask2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1]) # np.array([0,0,0,0,0,0,1,1,1,1,1,1])

    log_Y_pred1 = []
    log_Y_pred2 = []
    log_R_states1 = []
    log_R_states2 = []
    log_coupling = []

    for cs in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: # coupling strengths
        # print(f'cs: {cs}')

        ypred1 = []
        ypred2 = []
        rstates1 = []
        rstates2 = []
        coupling = np.array([cs] * len(np.vstack(X_test)))

        for x in X_test:
            y1 = np.array([np.zeros(readout1.output_dim)] * len(x))
            y2 = np.array([np.zeros(readout2.output_dim)] * len(x))
            r1 = np.array([np.zeros(reservoir1.output_dim)] * len(x))
            r2 = np.array([np.zeros(reservoir2.output_dim)] * len(x))

            for t in range(len(x)):
                noise = ns * np.random.randn(*x[t].shape)

                if t==0:
                    input_fb1 = np.concatenate(((noise + x[t]) * mask1, readout1.zero_state()), axis=None)
                    input_fb2 = np.concatenate(((noise + x[t]) * mask2, readout2.zero_state()), axis=None)
                else:
                    input_fb1 = np.concatenate(((noise + x[t]) * mask1, fbs * np.average([readout1.state(), readout2.state()], axis=0, weights=[(1-cs), cs])), axis=None)
                    input_fb2 = np.concatenate(((noise + x[t]) * mask2, fbs * np.average([readout1.state(), readout2.state()], axis=0, weights=[cs, (1-cs)])), axis=None)

                state1 = reservoir1.run(input_fb1)
                state2 = reservoir2.run(input_fb2)

                # predictions
                y1[t] = readout1.run(state1)
                y2[t] = readout2.run(state2)

                # save reservoir states
                r1[t] = state1
                r2[t] = state2

            # log
            ypred1.append(y1)
            ypred2.append(y2)
            rstates1.append(r1)
            rstates2.append(r2)

            # reset reservoirs
            reservoir1.reset(np.random.random(reservoir1.state().shape)) # reservoir1.reset()
            reservoir2.reset(np.random.random(reservoir2.state().shape)) # reservoir2.reset()

        # log over cs

        log_Y_pred1.append(ypred1)
        log_Y_pred2.append(ypred2)
        log_R_states1.append(rstates1)
        log_R_states2.append(rstates2)
        log_coupling.append(coupling)

    log_Y_pred1 = [data for c in log_Y_pred1 for data in c]
    log_Y_pred2 = [data for c in log_Y_pred2 for data in c]
    log_R_states1 = [data for c in log_R_states1 for data in c]
    log_R_states2 = [data for c in log_R_states2 for data in c]
    log_coupling = [data for c in log_coupling for data in c]

    return log_R_states1, log_R_states2, log_Y_pred1, log_Y_pred2, log_coupling


## IMPORT DATA ##


X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

X_train = X_train[:90]
Y_train = Y_train[:90]
X_test = X_test[:154]
Y_test = Y_test[:154]

# 30 test exemplars per class (same as training)
del X_test[30]
del Y_test[30]

del X_test[60:65]
del Y_test[60:65]

del X_test[90:]
del Y_test[90:]

# use speakers 2 and 3
del X_train[:30]
del Y_train[:30]

del X_test[:30]
del Y_test[:30]


# # check that lists are aligned
# xlist = [len(item) for item in X_test]
# ylist = [len(item) for item in Y_test]
# 
# if xlist == ylist:
#     print("lists are identical")
# else:
#     [i for i, (a, b) in enumerate(zip(xlist, ylist)) if a != b]


# 2-bit one-hot encodings
for i in range(len(Y_train)):
    Y_train[i] = Y_train[i][:,1:3]

for i in range(len(Y_test)):
    Y_test[i] = Y_test[i][:,1:3]


## HYPERPARAMS ##


fbs=1.0         # feedback scale factor
ns=0.0          # noise scale factor
r1_nnodes=100
r2_nnodes=100


## RUN ##


if __name__ == "__main__":
    runs = 8

    # create a multiprocessing pool
    with mp.Pool() as pool:      
        results = pool.map(run_models, range(runs))

    # compile results
    R_states1 = []
    R_states2 = []
    Y_pred1 = []
    Y_pred2 = []

    for i in range(runs):
        R_states1.append(results[i][0])
        R_states2.append(results[i][1])
        Y_pred1.append(results[i][2])
        Y_pred2.append(results[i][3])

    # store data
    R_states1 = [series for run in R_states1 for series in run]
    R_states2 = [series for run in R_states2 for series in run]
    Y_pred1 = [series for run in Y_pred1 for series in run]
    Y_pred2 = [series for run in Y_pred2 for series in run]

    Y1 = pd.DataFrame(np.vstack(Y_pred1))
    Y2 = pd.DataFrame(np.vstack(Y_pred2))
    R1 = pd.DataFrame(np.vstack(R_states1))
    R2 = pd.DataFrame(np.vstack(R_states2))

    Y1.to_csv(f'data/v2_s23_randreset_batch5/Y1_ns={ns}_r1={r1_nnodes}_r2={r2_nnodes}.csv', index=False)
    Y2.to_csv(f'data/v2_s23_randreset_batch5/Y2_ns={ns}_r1={r1_nnodes}_r2={r2_nnodes}.csv', index=False)
    R1.to_csv(f'data/v2_s23_randreset_batch5/R1_ns={ns}_r1={r1_nnodes}_r2={r2_nnodes}.csv', index=False)
    R2.to_csv(f'data/v2_s23_randreset_batch5/R2_ns={ns}_r1={r1_nnodes}_r2={r2_nnodes}.csv', index=False)

    targets = np.tile(np.concatenate(Y_test), (runs*11,1))

    meta_data = {
        'coupling': np.concatenate(np.tile(results[0][4], (runs,1))),
        'signal_idx': np.tile(np.concatenate([np.full(signal.shape[0], i+1) for i, signal in enumerate(X_test)]), runs*11),
        'run_idx': np.repeat(range(runs), len(results[0][4])),
        'target_n1': targets[:,0],
        'target_n2': targets[:,1],
    }

    meta_data = pd.DataFrame(meta_data)
    meta_data.to_csv(f'data/v2_s23_randreset_batch5/meta_data.csv', index=False)
