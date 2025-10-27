import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels

rpy.verbosity(0)


### Hyperparameters ###

nnodes = 500
warmup = 0
teacherfb_sigma = 0.2
feedback = False


### Load data ###

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)


### TODO 1 ###

# filter data s.t.
### signals shorter than 10 timesteps are removed,
### remaining signals are truncated to shortest length, and 
### an equal number of signals in each class remain
### explore the effect of truncating to beginning, middle, or end of signals


### TODO 2 ###

# using filtered dataset, train ESN (no feedback) on one timestep per signal for each timestep
### stop when there are no longer 9 classes represented
### plot classification accuracy for each timestep


### Initialize ESN layers ###

reservoir = Reservoir(nnodes, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)


### Train ###

train_states = []
train_targets = []

for (x, y) in zip(X_train, Y_train):
    for t in range(len(x)):
        # create input + feedback
        if t==0 or not feedback:
            fb = np.array(np.zeros(len(y[t]))) # no feedback on first timestep (or feedback==False)
            input = np.concatenate((x[t], fb), axis=None)
        else:
            fb = (y[t] + (teacherfb_sigma * np.random.randn(len(y[t])))) # teacher feedback + simulated Gaussian noise
            input = np.concatenate((x[t], fb), axis=None)
        
        # harvest reservoir states
        rstate = reservoir.run(input)

        if t >= warmup:
            train_states.append(rstate)
            train_targets.append(y[t, np.newaxis])
    
    # reset reservoir
    reservoir.reset()

# fit readout layer
readout.fit(train_states, train_targets)


### Test ###

Y_pred = []

for x in X_test:
    y = np.array([np.zeros(readout.output_dim)] * len(x))

    for t in range(len(x)):
        if t==0 or not feedback:
            fb = np.array(np.zeros(readout.output_dim)) # no feedback on first timestep (or feedback==False)
            input = np.concatenate((x[t], fb), axis=None)
        else:
            input = np.concatenate((x[t], readout.state()), axis=None)

        # harvest reservoir states + predictions
        rstate = reservoir.run(input)
        pred = readout.run(rstate)

        # store
        y[t] = pred

    # store
    Y_pred.append(y)

    # reset reservoir
    reservoir.reset()


# integrate readouts over time and select most active node as decision
acc = [np.argmax(target[0]) == np.argmax(np.sum(pred, axis=0)) for (target, pred) in zip(Y_test, Y_pred)]
print(f"Pcorrect: {np.mean(acc)}")


### TODO 3 ###

# plot readout values on each signal per class
### x axis: time
### y value: activation value (m + se, averaged over at least 10 simulations)
### color/fill: node
### save plots as .png
