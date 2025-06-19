import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge, FORCE, RLS
from reservoirpy.datasets import to_forecasting
from reservoirpy.datasets import mackey_glass

### Step-by-step tutorial ###

# reservoir = Reservoir(100, lr=0.5, sr=0.9)
# readout = Ridge(ridge=1e-7)
# 
# X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
# 
# X_train = X[:50]
# Y_train = X[1:51]
# 
# plt.figure(figsize=(10, 3))
# plt.title("A sine wave and its future.")
# plt.xlabel("$t$")
# plt.plot(X_train, label="sin(t)", color="blue")
# plt.plot(Y_train, label="sin(t+1)", color="red")
# plt.legend()
# plt.show()
# 
# train_states = reservoir.run(X_train, reset=True)
# readout = readout.fit(train_states, Y_train, warmup=10)
# 
# test_states = reservoir.run(X[50:])
# Y_pred = readout.run(test_states)
# 
# plt.figure(figsize=(10, 3))
# plt.title("A sine wave and its future.")
# plt.xlabel("$t$")
# plt.plot(Y_pred, label="Predicted sin(t)", color="blue")
# plt.plot(X[51:], label="Real sin(t+1)", color="red")
# plt.legend()
# plt.show()



### Create ESN model tutorial ###

# X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
# 
# X_train = X[:50]
# Y_train = X[1:51]
# 
# reservoir = Reservoir(100, lr=0.5, sr=0.9)
# readout = Ridge(ridge=1e-7)
# 
# esn_model = reservoir >> readout
# 
# esn_model = esn_model.fit(X_train, Y_train, warmup=10)
# print(reservoir.is_initialized, readout.is_initialized, readout.fitted)
# 
# Y_pred = esn_model.run(X[50:])
# 
# plt.figure(figsize=(10, 3))
# plt.title("A sine wave and its future.")
# plt.xlabel("$t$")
# plt.plot(Y_pred, label="Predicted sin(t+1)", color="blue")
# plt.plot(X[51:], label="Real sin(t+1)", color="red")
# plt.legend()
# plt.show()



### Feedback connection ###

# X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
# 
# X_train = X[:50]
# Y_train = X[1:51]
# 
# reservoir = Reservoir(100, lr=0.5, sr=0.9)
# readout = Ridge(ridge=1e-7)
# 
# reservoir <<= readout
# 
# esn_model = reservoir >> readout
# 
# esn_model = esn_model.fit(X_train, Y_train)
# 
# esn_model(X[0].reshape(1, -1))
# 
# print("Feedback received (reservoir):", reservoir.feedback())
# print("State sent: (readout):", readout.state())



### Online learning for chaotic timeseries forecasting ###

# # Helper functions
# 
# def plot_train_test(X_train, y_train, X_test, y_test):
#     sample = 500
#     test_len = X_test.shape[0]
#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
#     plt.plot(np.arange(0, 500), y_train[-sample:], label="Training ground truth")
#     plt.plot(np.arange(500, 500+test_len), X_test, label="Testing data")
#     plt.plot(np.arange(500, 500+test_len), y_test, label="Testing ground truth")
#     plt.legend()
#     plt.show()
# 
# def plot_results(y_pred, y_test, sample=500):
# 
#     fig = plt.figure(figsize=(15, 7))
#     plt.subplot(211)
#     plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
#     plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
#     plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
# 
#     plt.legend()
#     plt.show()
# 
# # Hyperparameters
# 
# units = 100
# leak_rate = 0.3
# spectral_radius = 1.25
# input_scaling = 1.0
# connectivity = 0.1
# input_connectivity = 0.2
# seed = 1234
# 
# # Curate data
# 
# timesteps = 2510
# tau = 17
# X = mackey_glass(timesteps, tau=tau)
# 
# X = 2 * (X - X.min()) / (X.max() - X.min()) - 1 # rescale between -1 and 1
# 
# x, y = to_forecasting(X, forecast=10)
# X_train1, y_train1 = x[:2000], y[:2000]
# X_test1, y_test1 = x[2000:], y[2000:]
# 
# plot_train_test(X_train1, y_train1, X_test1, y_test1)
# 
# # Train ESN
# 
# reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
#                       lr=leak_rate, rc_connectivity=connectivity,
#                       input_connectivity=input_connectivity, seed=seed)
# 
# readout   = RLS(alpha=1e-1) # FORCE(1)
# 
# esn_online = reservoir >> readout
# 
# outputs_pre = np.zeros(X_train1.shape)
# for t, (x, y) in enumerate(zip(X_train1, y_train1)): # for each timestep of training data:
#     outputs_pre[t, :] = esn_online.train(x, y)
# 
# plot_results(outputs_pre, y_train1, sample=100)
# 
# plot_results(outputs_pre, y_train1, sample=500)



### Online learning with RLS ###

# x = np.random.normal(size=(100, 3))
# noise = np.random.normal(scale=0.1, size=(100, 1))
# y = x @ np.array([[10], [-0.2], [7]]) + noise + 12
# 
# plt.figure(figsize=(10, 3))
# plt.xlabel("$t$")
# plt.plot(x, label="input", color="blue")
# plt.plot(y, label="target", color="red")
# plt.legend()
# plt.show()
# 
# rls_node = RLS(alpha=1e-1)
# 
# _ = rls_node.train(x[:5], y[:5])
# print(rls_node.Wout.T, rls_node.bias)
# 
# _ = rls_node.train(x[5:], y[5:])
# print(rls_node.Wout.T, rls_node.bias)



### Online learning with feedback connection ###

# ## Case #1:

# # Helper functions
# 
# def plot_train_test(X_train, y_train, X_test, y_test):
#     sample = 500
#     test_len = X_test.shape[0]
#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
#     plt.plot(np.arange(0, 500), y_train[-sample:], label="Training ground truth")
#     plt.plot(np.arange(500, 500+test_len), X_test, label="Testing data")
#     plt.plot(np.arange(500, 500+test_len), y_test, label="Testing ground truth")
#     plt.legend()
#     plt.show()
# 
# def plot_results(y_pred, y_test, sample=500):
# 
#     fig = plt.figure(figsize=(15, 7))
#     plt.subplot(211)
#     plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
#     plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
#     plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
# 
#     plt.legend()
#     plt.show()
# 
# # Hyperparameters
# 
# units = 100
# leak_rate = 0.3
# spectral_radius = 1.25
# input_scaling = 1.0
# connectivity = 0.1
# input_connectivity = 0.2
# seed = 1234
# 
# # Curate data
# 
# timesteps = 2510
# tau = 17
# X = mackey_glass(timesteps, tau=tau)
# 
# X = 2 * (X - X.min()) / (X.max() - X.min()) - 1 # rescale between -1 and 1
# 
# x, y = to_forecasting(X, forecast=10)
# X_train1, y_train1 = x[:2000], y[:2000]
# X_test1, y_test1 = x[2000:], y[2000:]
# 
# plot_train_test(X_train1, y_train1, X_test1, y_test1)
# 
# # Train ESN
# 
# reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
#                       lr=leak_rate, rc_connectivity=connectivity,
#                       input_connectivity=input_connectivity, seed=seed)
# 
# readout   = RLS(alpha=1e-1)
# 
# reservoir <<= readout
# 
# esn_online = reservoir >> readout
# 
# outputs_pre = np.zeros(X_train1.shape)
# for t, (x, y) in enumerate(zip(X_train1, y_train1)): # for each timestep of training data:
#     outputs_pre[t, :] = esn_online.train(x, y)
# 
# plot_results(outputs_pre, y_train1, sample=100)
# 
# plot_results(outputs_pre, y_train1, sample=500)

# ## Case #2:
# 
# X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
# 
# X_train = X[:50]
# Y_train = X[1:51]
# 
# reservoir = Reservoir(100, lr=0.5, sr=0.9)
# readout = RLS(alpha=1e-1) # Ridge(ridge=1e-7)
# 
# reservoir <<= readout
# 
# esn_model = reservoir >> readout
# 
# outputs_pre = np.zeros(X_train.shape)
# for t, (x, y) in enumerate(zip(X_train, Y_train)): # for each timestep of training data:
#     outputs_pre[t, :] = esn_model.train(x, y)
#     # print(reservoir.feedback() == readout.state()) # confirm that feedback is sent correctly
# 
# plt.figure(figsize=(10, 3))
# plt.title("A sine wave and its future.")
# plt.xlabel("$t$")
# plt.plot(outputs_pre, label="Predicted sin(t+1)", color="blue")
# plt.plot(Y_train, label="Real sin(t+1)", color="red")
# plt.legend()
# plt.show()

