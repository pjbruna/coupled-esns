import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from reservoirpy import set_seed, verbosity
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat
from reservoirpy.observables import rmse, rsquare
from reservoirpy.datasets import japanese_vowels

verbosity(0)

## DATA ##

# X_train, Y_train, X_test, Y_test = japanese_vowels()
# 
# plt.figure()
# plt.imshow(X_train[0].T, vmin=-1.2, vmax=2)
# plt.title(f"A sample vowel of speaker {np.argmax(Y_train[0]) +1}")
# plt.xlabel("Timesteps")
# plt.ylabel("LPC (cepstra)")
# plt.colorbar()
# plt.show()
# 
# plt.figure()
# plt.imshow(X_train[50].T, vmin=-1.2, vmax=2)
# plt.title(f"A sample vowel of speaker {np.argmax(Y_train[50]) +1}")
# plt.xlabel("Timesteps")
# plt.ylabel("LPC (cepstra)")
# plt.colorbar()
# plt.show()
# 
# sample_per_speaker = 30
# n_speaker = 9
# X_train_per_speaker = []
# 
# for i in range(n_speaker):
#     X_speaker = X_train[i*sample_per_speaker: (i+1)*sample_per_speaker]
#     X_train_per_speaker.append(np.concatenate(X_speaker).flatten())
# 
# plt.boxplot(X_train_per_speaker)
# plt.xlabel("Speaker")
# plt.ylabel("LPC (cepstra)")
# plt.show()


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


source = Input()
reservoir = Reservoir(100, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

reservoir << readout

model = source >> reservoir >> readout

Y_pred = model.fit(X_train, Y_train, stateful=False, warmup=2).run(X_test, stateful=False)

Y_pred_class = [np.argmax(y_p, axis=1) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t, axis=1) for y_t in Y_test]

score = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(Y_pred_class, axis=0))

print("Accuracy: ", f"{score * 100:.3f} %")

# # Save
# 
# Y1 = pd.DataFrame(np.vstack(Y_pred))
# Y2 = pd.DataFrame(np.vstack(Y_test))
# 
# Y1.to_csv(f'data/baseline/Ypred.csv')
# Y2.to_csv(f'data/baseline/Ytest.csv')
# 
# signal_idx = pd.DataFrame(np.concatenate([np.full(signal.shape[0], i+1) for i, signal in enumerate(X_test)]))
# signal_idx.to_csv(f'data/baseline/signal_idx.csv')

