import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from reservoirpy import set_seed, verbosity
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat
from reservoirpy.observables import rmse, rsquare
from reservoirpy.datasets import japanese_vowels

## HELPER FUNCTIONS ##


# Train w/ feedback

def train(reservoir=None, readout=None, input=None, target=None, feedback_scaling=None):

    # Create feedback signal

    fb_scaling = feedback_scaling
    fb_signal = [signal * fb_scaling for signal in target]

    for i in range(len(fb_signal)):
        fb_signal[i][0] = np.array([[0] * len(fb_signal[i][0])]) #  mask first timestep of each signal

    # Train

    train_states = []
    for (x, y) in zip(input, fb_signal):
        input_fb = Concat().run((x, y))
        train_states.append(reservoir.run(input_fb))
        reservoir.reset()

    readout.fit(train_states, target)

    return reservoir, readout


# Test w/ feedback

def test(reservoir=None, readout=None, input=None, feedback_scaling=None, noise_scaling=None):
    Y_pred = []
    R_states = []

    for x in input:
        y = np.array([np.zeros(readout.output_dim)] * len(x))
        r = np.array([np.zeros(reservoir.output_dim)] * len(x))

        for t in range(len(x)):
            noise = noise_scaling * np.random.randn(*x[t].shape)

            if t==0:
                input_fb = Concat().run((noise + x[t], readout.zero_state()))
            else:
                input_fb = Concat().run((noise + x[t], feedback_scaling * readout.state()))

            state = reservoir.run(input_fb)
            y[t] = readout.run(state)
            r[t] = state

        Y_pred.append(y)
        R_states.append(r)
        reservoir.reset()
    
    return Y_pred, R_states


# Calculate accuracy

# def evaluate(predictions1=None, predictions2=None, target=None, endpoint=False):
#     if endpoint==True: # calculate accuracy based only on final timestep
#         predictions1 = [signal[-1] for signal in predictions1]
#         predictions2 = [signal[-1] for signal in predictions2]
#         target = [signal[-1] for signal in target]
# 
#         pred_class1 = [np.argmax(y_p) for y_p in predictions1]
#         pred_class2 = [np.argmax(y_p) for y_p in predictions2]
#         Y_test_class = [np.argmax(y_t) for y_t in target]
# 
#         score1 = accuracy_score(Y_test_class, pred_class1)
#         score2 = accuracy_score(Y_test_class, pred_class2)
#     else:
#         pred_class1 = [np.argmax(y_p, axis=1) for y_p in predictions1]
#         pred_class2 = [np.argmax(y_p, axis=1) for y_p in predictions2]
#         Y_test_class = [np.argmax(y_t, axis=1) for y_t in target]
# 
#         score1 = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(pred_class1, axis=0))
#         score2 = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(pred_class2, axis=0))
# 
#     return score1, score2


def evaluate(predictions=None, target=None, endpoint=False):
    if endpoint==True: # calculate accuracy based only on final timestep
        target = [signal[-1] for signal in target]
        Y_test_class = [np.argmax(y_t) for y_t in target]

        scores= []
        for responses in predictions:
            pred = [signal[-1] for signal in responses]
            pred_class = [np.argmax(y_p) for y_p in pred]
            score = accuracy_score(Y_test_class, pred_class)
            scores.append(score)
        
    else:
        Y_test_class = [np.argmax(y_t, axis=1) for y_t in target]

        scores = []
        for responses in predictions:
            pred_class = [np.argmax(y_p, axis=1) for y_p in responses]
            score = accuracy_score(np.concatenate(Y_test_class, axis=0), np.concatenate(pred_class, axis=0))
            scores.append(score)

    return scores


# Plot readout over time

def plot_readout(data=None, title=None, save=False):
    plt.figure(figsize=(8, 6))

    for i in range(len(data)):
        signal = data[i]
        timesteps = np.arange(len(signal)) / (len(signal) - 1)
 
        x = signal[:,0]
        y = signal[:,1]
 
        scatter = plt.scatter(x, y, s=20, marker='x', c=timesteps, cmap='viridis')

    plt.title(f'{title}')
    plt.xlabel('$R_1$')
    plt.ylabel('$R_2$')
    plt.axline((0,0), slope=1, color='black', linestyle='dotted')
    plt.colorbar(scatter, label='Normalized Time')
    plt.grid(True)
    # plt.legend()

    if save==False:
        # plt.show()
        plt.clf()
    else:
        plt.savefig(save)


# Plot decision over time

def plot_timeseries(data=None, target=None, title=None, save=False):
    plt.figure(figsize=(8, 6))

    for i in range(len(data)):
        signal = data[i]
        x = signal[:,0]
        y = signal[:,1]

        if np.mean(target[i][:,0], axis=0)==1:
            scatter = plt.plot(x - y, color='blue', alpha=0.3, label='Class A')
        else:
            scatter = plt.plot(x - y, color='orange', alpha=0.3, label='Class B')


    legend_handles = [
        mpatches.Patch(color='blue', label='Class A'),
        mpatches.Patch(color='orange', label='Class B')
    ]

    plt.title(f'{title}')
    plt.xlabel('$t$')
    plt.ylabel('$R_1$ - $R_2$')
    plt.hlines(0, xmin=plt.gca().get_xlim()[0], xmax=plt.gca().get_xlim()[1], color='black', linestyle='dotted')
    plt.grid(True)
    plt.legend(handles=legend_handles, loc='upper right')
    
    if save==False:
        # plt.show()
        plt.clf()
    else:
        plt.savefig(save)


# Plot PCA

def plot_pca(data=None, targets=None, series=None, title=None, save=False):
    res_states = np.vstack(data)
    
    pca = PCA(n_components=2)
    pca.fit(res_states)

    transformed_signals = [pca.transform(signal) for signal in data]

    plt.figure(figsize=(10, 7))

    for i, transformed_signals in enumerate(transformed_signals):
        if i not in series:
            continue
        else:
            timesteps = np.arange(len(transformed_signals)) / (len(transformed_signals) - 1)

            if np.mean(targets[i][:,0], axis=0)==1:
                m='o'
            else:
                m='x'

            x = transformed_signals[:,0]
            y = transformed_signals[:,1]

            plt.plot(x, y, color='gray', alpha=0.3)
            scatter = plt.scatter(x, y, s=20, c=timesteps, cmap='viridis', marker=m)

    plt.title(f'{title}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Normalized Time')
    # plt.legend()
    plt.grid(True)
    
    if save==False:
        # plt.show()
        plt.clf()
    else:
        plt.savefig(save)