import numpy as np
import pandas as pd
import reservoirpy as rpy
import multiprocessing as mp
# from sklearn.metrics import accuracy_score
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from functions import *

rpy.verbosity(0)


# Define model

def run_models(index):
    print(f'Running model: {index}')

    reservoir1 = Reservoir(units=500, sr=0.9, lr=0.1, activation='tanh')
    readout1 = Ridge(ridge=1e-6)

    # Train
    train_states1 = []
    train_targets = []

    for (x, y) in zip(X_train, Y_train):
        for t in range(len(x)):
            if t==0:
                input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
            else:
                input_fb = np.concatenate((x[t], y[t]+(np.random.randn(len(y[t])) * 0.2)), axis=None)

            rstate = reservoir1.run(input_fb)
            train_states1.append(rstate)
            train_targets.append(y[t, np.newaxis])

        # reset reservoirs
        reservoir1.reset(np.random.random(reservoir1.state().shape)) # reservoir1.reset()

    readout1.fit(train_states1, train_targets)

    # Test
    Y_pred = []
    for x in X_test:
        r1 = np.array([np.zeros(reservoir1.output_dim)] * len(x))
        y1 = np.array([np.zeros(readout1.output_dim)] * len(x))

        for t in range(len(x)):
            if t==0:
                input_fb = np.concatenate((x[t], np.array(np.zeros(readout1.output_dim))), axis=None)
            else:
                input_fb = np.concatenate((x[t], readout1.state()), axis=None)

            rstate = reservoir1.run(input_fb)
            ypred = readout1.run(rstate)

            r1[t] = rstate
            y1[t] = ypred

        # log predictions
        Y_pred.append(y1)

        # reset reservoirs
        reservoir1.reset(np.random.random(reservoir1.state().shape)) # reservoir1.reset()


    # Performance

    persig_rmse = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, Y_pred)]
    
    print(f'RMSE: {np.mean(persig_rmse)}')

    # Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    # Y_test_class = [np.argmax(y_t) for y_t in Y_test]
    # 
    # score = accuracy_score(Y_test_class, Y_pred_class)
    # 
    # print("Accuracy: ", f"{score * 100:.3f} %")

    return Y_pred, Y_test


# Import data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# Run in parallel

if __name__ == "__main__":
    runs = 8

    # create a multiprocessing pool
    with mp.Pool() as pool:      
        results = pool.map(run_models, range(runs))

        # compile results
        predictions = []
        targets = []

        for i in range(runs):
            predictions.append(results[i][0])
            targets.append(results[i][1])

        # store data
        predictions = [series for run in predictions for series in run]
        targets = [series for run in targets for series in run]

        Y = pd.DataFrame(np.vstack(predictions))
        T = pd.DataFrame(np.vstack(targets))

        Y.to_csv(f'data/esn_fb/predictions.csv', index=False)
        T.to_csv(f'data/esn_fb/targets.csv', index=False)

        meta_data = {
            'signal_idx': np.tile(np.concatenate([np.full(signal.shape[0], i+1) for i, signal in enumerate(X_test)]), runs),
            'run_idx': np.repeat(range(runs), len(np.vstack(results[0][0]))),
        }

        meta_data = pd.DataFrame(meta_data)
        meta_data.to_csv(f'data/esn_fb/meta_data.csv', index=False)
