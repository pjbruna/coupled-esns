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

    reservoir2 = Reservoir(units=500, sr=0.9, lr=0.1, activation='tanh')
    readout2 = Ridge(ridge=1e-6)

    # Train

    train_targets = []
    train_states1 = []
    train_states2 = []

    for (x, y) in zip(X_train, Y_train):
        for t in range(len(x)):
            if t==0:
                input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
            else:
                input_fb = np.concatenate((x[t], y[t]+(np.random.randn(len(y[t])) * 0.2)), axis=None)

            rstate1 = reservoir1.run(input_fb)
            rstate2 = reservoir2.run(input_fb)

            train_states1.append(rstate1)
            train_states2.append(rstate2)

            train_targets.append(y[t, np.newaxis])

        # reset reservoirs
        reservoir1.reset()
        reservoir2.reset()

    readout1.fit(train_states1, train_targets)
    readout2.fit(train_states2, train_targets)

    # Test

    coupling = []
    R1_states = []
    R2_states = []
    Y1_preds = []
    Y2_preds = []

    for cs in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: # coupling strengths

        R_states1 = []
        R_states2 = []
        Y_pred1 = []
        Y_pred2 = []

        coupling.append(np.array([cs] * len(np.vstack(X_test))))

        for x in X_test:
            r1 = np.array([np.zeros(reservoir1.output_dim)] * len(x))
            r2 = np.array([np.zeros(reservoir2.output_dim)] * len(x))
            y1 = np.array([np.zeros(readout1.output_dim)] * len(x))
            y2 = np.array([np.zeros(readout2.output_dim)] * len(x))

            for t in range(len(x)):
                if t==0:
                    input_fb1 = np.concatenate((x[t], np.array(np.zeros(readout1.output_dim))), axis=None)
                    input_fb2 = input_fb1
                else:
                    input_fb1 = np.concatenate(((x[t]), np.average([readout1.state(), readout2.state()], axis=0, weights=[(1-cs), cs])), axis=None)
                    input_fb2 = np.concatenate(((x[t]), np.average([readout1.state(), readout2.state()], axis=0, weights=[cs, (1-cs)])), axis=None)

                rstate1 = reservoir1.run(input_fb1)
                rstate2 = reservoir2.run(input_fb2)

                ypred1 = readout1.run(rstate1)
                ypred2 = readout2.run(rstate2)

                # store
                r1[t] = rstate1
                r2[t] = rstate2
                y1[t] = ypred1
                y2[t] = ypred2

            # store
            R_states1.append(r1)
            R_states2.append(r2)
            Y_pred1.append(y1)
            Y_pred2.append(y2)

            # reset reservoirs
            reservoir1.reset()
            reservoir2.reset()


        # Performance

        persig_rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, Y_pred1)]
        persig_rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, Y_pred2)]

        print(f'CS: {cs}; RMSE1: {np.mean(persig_rmse1)}; RMSE2: {np.mean(persig_rmse2)}')

        R1_states.append(R_states1)
        R2_states.append(R_states2)
        Y1_preds.append(Y_pred1)
        Y2_preds.append(Y_pred2)

    coupling = [data for c in coupling for data in c]
    R1_states = [data for c in R1_states for data in c]
    R2_states = [data for c in R2_states for data in c]
    Y1_preds = [data for c in Y1_preds for data in c]
    Y2_preds = [data for c in Y2_preds for data in c]

    return R1_states, R2_states, Y1_preds, Y2_preds, coupling


# Import data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

# Run in parallel

if __name__ == "__main__":
    runs = 8

    # create a multiprocessing pool
    with mp.Pool() as pool:      
        results = pool.map(run_models, range(runs))

        # compile results
        R1 = []
        R2 = []
        Y1 = []
        Y2 = []

        for i in range(runs):
            R1.append(results[i][0])
            R2.append(results[i][1])
            Y1.append(results[i][2])
            Y2.append(results[i][3])

        # store data
        R1 = [series for run in R1 for series in run]
        R2 = [series for run in R2 for series in run]
        Y1 = [series for run in Y1 for series in run]
        Y2 = [series for run in Y2 for series in run]

        R1_df = pd.DataFrame(np.vstack(R1))
        R2_df = pd.DataFrame(np.vstack(R2))
        Y1_df = pd.DataFrame(np.vstack(Y1))
        Y2_df = pd.DataFrame(np.vstack(Y2))

        R1_df.to_csv(f'data/a_new_hope/R1.csv', index=False)
        R2_df.to_csv(f'data/a_new_hope/R2.csv', index=False)
        Y1_df.to_csv(f'data/a_new_hope/Y1.csv', index=False)
        Y2_df.to_csv(f'data/a_new_hope/Y2.csv', index=False)

        targets = np.tile(np.concatenate(Y_test), (runs*11,1))

        meta_data = {
            'coupling': np.concatenate(np.tile(results[0][4], (runs,1))),
            'signal_idx': np.tile(np.concatenate([np.full(signal.shape[0], i+1) for i, signal in enumerate(X_test)]), runs*11),
            'run_idx': np.repeat(range(runs), len(results[0][4])),
            'TX0': targets[:,0],
            'TX1': targets[:,1],
            'TX2': targets[:,2],
            'TX3': targets[:,3],
            'TX4': targets[:,4],
            'TX5': targets[:,5],
            'TX6': targets[:,6],
            'TX7': targets[:,7],
            'TX8': targets[:,8],
        }

        meta_data = pd.DataFrame(meta_data)
        meta_data.to_csv(f'data/a_new_hope/meta_data.csv', index=False)




