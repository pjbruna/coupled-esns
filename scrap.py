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


## MODEL ##


readout = Ridge(ridge=1e-6)

readout.fit(X_train, X_train)

# print(f'Wout: {readout.Wout}')
# print(f'Bias: {readout.bias}')

out = readout.run(X_train)



# x = np.random.normal(size=(100, 3))
# noise = np.random.normal(scale=0.1, size=(100, 1))
# y = x @ np.array([[10], [-0.2], [7]]) + noise + 12
# 
# ridge_regressor = Ridge(ridge=1e-6)
# ridge_regressor.fit(x, y)
# 
# print(ridge_regressor.Wout)
# print(ridge_regressor.bias)