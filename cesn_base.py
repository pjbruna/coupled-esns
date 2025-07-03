import numpy as np
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *
from functions import *

rpy.verbosity(0)

# Hyperparams

runs = 10 # number of runs
coupling_num = 11 # number of coupling strengths to test
seed_num = None # seed reservoirs?

train_sample = 0.794 # training data size; range of interest: 0.5 (non-overlapping) -- 1 (completely overlapping)

r1_size = 153 # reservoir 1 size
r2_size = r1_size # int(r1_size / 2) # reservoir 2 size


# Curate training data

X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

sampled_indices1 = np.random.choice(len(X_train), size=int(len(X_train) * train_sample), replace=False)
X_train1 = [X_train[i] for i in sampled_indices1]
Y_train1 = [Y_train[i] for i in sampled_indices1]

if train_sample==0.5:
    sampled_indices2 = np.setdiff1d(np.arange(len(X_train)), sampled_indices1) # sample (non-overlapping)
    np.random.shuffle(sampled_indices2)
    X_train2 = [X_train[i] for i in sampled_indices2]
    Y_train2 = [Y_train[i] for i in sampled_indices2]
else:
    sampled_indices2 = np.random.choice(len(X_train), size=int(len(X_train) * train_sample), replace=False) # sample (random)
    X_train2 = [X_train[i] for i in sampled_indices2]
    Y_train2 = [Y_train[i] for i in sampled_indices2]


# Run model

store_couplings = []
store_accuracies_joint = []
store_accuracies_y1 = []
store_accuracies_y2 = []

for c in np.linspace(0.0, 1.0, num=coupling_num):
    c = round(c, 1)
    acc_joint = []
    acc_y1 = []
    acc_y2 = []

    for r in range(runs):
        print(f'Simulation #{r}')
        model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=c, is_seed=seed_num)

        model.train_r1(input=X_train1, target=Y_train1)
        model.train_r2(input=X_train2, target=Y_train2)

        results = model.test(input=X_test, target=Y_test, noise_scale=0, do_print=False, save_reservoir=False)

        joint, y1, y2 = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test, do_print=False)

        acc_joint.append(joint)
        acc_y1.append(y1)
        acc_y2.append(y2)

    store_couplings.append(c)
    store_accuracies_joint.append(acc_joint)
    store_accuracies_y1.append(acc_y1)
    store_accuracies_y2.append(acc_y2)
    
    print(f'N: {runs}; CS: {c}; R1: {r1_size}; R2: {r2_size};  Acc: {np.mean(acc_joint)}')


# Plot

if seed_num==None:
    # plot_coupling_strengths(x=store_couplings, y=store_accuracies_joint, do_print=False, save=f'plt_figs/acc_x_coupling_N={runs}_coupling={coupling_num}_R1={r1_size}_R2={r2_size}_sample={train_sample}.png')
    plot_coupling_with_comparison(x=store_couplings, y_joint=store_accuracies_joint, y1=store_accuracies_y1, y2=store_accuracies_y2, do_print=False, save=f'plt_figs/acc_with_comparison_N={runs}_coupling={coupling_num}_R1={r1_size}_R2={r2_size}_sample={train_sample}.png')
else:
    # plot_coupling_strengths(x=store_couplings, y=store_accuracies_joint, do_print=False, save=f'plt_figs_seed/acc_x_coupling_N={runs}_coupling={coupling_num}_R1={r1_size}_R2={r2_size}_sample={train_sample}.png')
    plot_coupling_with_comparison(x=store_couplings, y_joint=store_accuracies_joint, y1=store_accuracies_y1, y2=store_accuracies_y2, do_print=False, save=f'plt_figs_seed/acc_with_comparison_N={runs}_coupling={coupling_num}_R1={r1_size}_R2={r2_size}_sample={train_sample}.png')




#############################



# rpy.verbosity(0)
# 
# # Hyperparams
# 
# nnodes = 300 # reservoir size
# n = 0 # noise
# cs = 0 # coupling strength
# 
# # Import data
# 
# X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)
#     
# # ESNs
# 
# reservoir1 = Reservoir(units=nnodes, sr=0.9, lr=0.1, activation='tanh') # seed=42
# readout1 = Ridge(ridge=1e-6)
# 
# reservoir2 = Reservoir(units=nnodes, sr=0.9, lr=0.1, activation='tanh') # seed=42
# readout2 = Ridge(ridge=1e-6)
# 
# # Train
# 
# train_targets = []
# train_states1 = []
# train_states2 = []
# 
# for (x, y) in zip(X_train, Y_train):
#     # final = len(x)-1
# 
#     for t in range(len(x)):
#         if t==0:
#             input_fb = np.concatenate((x[t], np.array(np.zeros(len(y[t])))), axis=None)
#         else:
#             input_fb = np.concatenate((x[t], y[t]+(np.random.randn(len(y[t])) * 0.2)), axis=None)
# 
#         rstate1 = reservoir1.run(input_fb)
#         rstate2 = reservoir2.run(input_fb)
# 
#         # if t==final:
#         #     train_states1.append(rstate1)
#         #     train_states2.append(rstate2)
#         #     train_targets.append(y[t, np.newaxis])
# 
#         train_states1.append(rstate1)
#         train_states2.append(rstate2)
#         train_targets.append(y[t, np.newaxis])
# 
#     # reset reservoirs
#     reservoir1.reset()
#     reservoir2.reset()
# 
# readout1.fit(train_states1, train_targets)
# readout2.fit(train_states2, train_targets)
# 
# # Test
# 
# Y_pred1 = []
# Y_pred2 = []
# 
# for x in X_test:
#     y1 = np.array([np.zeros(readout1.output_dim)] * len(x))
#     y2 = np.array([np.zeros(readout2.output_dim)] * len(x))
# 
#     for t in range(len(x)):
#         if t==0:
#             input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * n)), np.array(np.zeros(readout1.output_dim))), axis=None)
#             input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * n)), np.array(np.zeros(readout2.output_dim))), axis=None)
#         else:
#             input_fb1 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * n)), np.average([readout1.state(), readout2.state()], axis=0, weights=[(1-cs), cs])), axis=None)
#             input_fb2 = np.concatenate(((x[t] + (np.random.randn(len(x[t])) * n)), np.average([readout1.state(), readout2.state()], axis=0, weights=[cs, (1-cs)])), axis=None)
#         
#         rstate1 = reservoir1.run(input_fb1)
#         rstate2 = reservoir2.run(input_fb2)
# 
#         ypred1 = readout1.run(rstate1)
#         ypred2 = readout2.run(rstate2)
# 
#         # store
#         y1[t] = ypred1
#         y2[t] = ypred2
# 
#     # store
#     Y_pred1.append(y1)
#     Y_pred2.append(y2)
# 
#     # reset reservoirs
#     reservoir1.reset()
#     reservoir2.reset()
# 
# # Performance
# 
# Y_pred1 = [series[-1] for series in Y_pred1]
# Y_pred2 = [series[-1] for series in Y_pred2]
# Y_test = [series[-1] for series in Y_test]
# 
# acc1 = [np.argmax(test) == np.argmax(pred1) for (test, pred1) in zip(Y_test, Y_pred1)]
# acc2 = [np.argmax(test) == np.argmax(pred2) for (test, pred2) in zip(Y_test, Y_pred2)]
# 
# print(f'CS: {cs}; Acc1: {np.mean(acc1)}; Acc2: {np.mean(acc2)}')
# 
# # persig_rmse1 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, Y_pred1)]
# # persig_rmse2 = [np.mean(np.sqrt(np.mean((test - pred)**2, axis=1))) for (test, pred) in zip(Y_test, Y_pred2)]
# # 
# # print(f'CS: {cs}; RMSE1: {np.mean(persig_rmse1)}; RMSE2: {np.mean(persig_rmse2)}')
# # 
# # acc1 = [np.argmax(test, axis=1) == np.argmax(pred1, axis=1) for (test, pred1) in zip(Y_test, Y_pred1)]
# # acc2 = [np.argmax(test, axis=1) == np.argmax(pred2, axis=1) for (test, pred2) in zip(Y_test, Y_pred2)]
# # 
# # print(f'CS: {cs}; Acc1: {np.mean(np.concatenate(acc1))}; Acc2: {np.mean(np.concatenate(acc2))}')
