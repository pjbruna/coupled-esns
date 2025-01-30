import numpy as np
import reservoirpy as rpy
from sklearn.metrics import accuracy_score
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.datasets import japanese_vowels
from functions import *

rpy.verbosity(0)

X_train, Y_train, X_test, Y_test = japanese_vowels()

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

model = source >> reservoir >> readout

states_train = []
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states[-1, np.newaxis])

readout.fit(states_train, Y_train)

Y_pred = []
for x in X_test:
    states = reservoir.run(x, reset=True)
    y = readout.run(states[-1, np.newaxis])
    Y_pred.append(y)

sqrloss = [(test - pred)**2 for (test, pred) in zip(Y_test, Y_pred)]
rmse = np.sqrt(np.mean(np.mean(sqrloss, axis=1)))

print(f'RMSE; {rmse}')

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t) for y_t in Y_test]

score = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")
