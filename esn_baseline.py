import numpy as np
import reservoirpy as rpy
from sklearn.metrics import accuracy_score
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.datasets import japanese_vowels
import matplotlib.pyplot as plt

rpy.verbosity(0)


# Load data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)


# Initialize ESN layers

reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)


# Train

states_train = []
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states)

readout.fit(states_train, Y_train)


# Test

Y_pred = []
for x in X_test:
    states = reservoir.run(x, reset=True)
    y = readout.run(states)
    Y_pred.append(y)

sqrloss = [(test - pred)**2 for (test, pred) in zip(Y_test, Y_pred)]
rmse = np.sqrt(np.mean(np.mean(sqrloss, axis=1)))

print(f'RMSE; {rmse}')


# Score

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t) for y_t in Y_test]

score = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")


# Plot decision trajectories

signals_by_class = {i: [] for i in range(9)}
for i, (y_true, y_seq) in enumerate(zip(Y_test_class, Y_pred)):
    signals_by_class[y_true].append(y_seq)

for category, signals in signals_by_class.items():
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(10, 2*n_signals), sharex=True, sharey=True)
    
    if n_signals == 1:
        axes = [axes]  # ensure iterable if only one signal
    
    for idx, (ax, signal) in enumerate(zip(axes, signals)):
        for readout_idx in range(9):
            ax.plot(signal[:, readout_idx], label=f"Readout {readout_idx+1}")
        ax.set_title(f"Category {category} - Signal {idx+1}")
        ax.set_ylabel("Activation")
    
    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Readout activations over time for category {category}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()