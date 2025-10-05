import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from data_processing import *
from cesn_model import *
 

# seed = 42
# np.random.seed(seed)
# reservoir_seeds = [seed,seed]

save_data = True
rpy.verbosity(0)


# Hyperparams
 
runs = 50
siglen = 10 # filter and truncate train/test signals
noise = [0.4,0.4] # noise added to input signals during testing (equal vs unequal conditions)
rsize = 500
reset_state = 'zero' # reset reservoirs between signals


# Run model

store_coupling = []
store_accuracies_joint = []
store_accuracies_upper = []
store_accuracies_lower = []
store_accuracies_avg = []

init=True

for c in ["independent", "interaction", "integration"]:
    acc_joint = []
    acc_upper = []
    acc_lower = []
    acc_avg = []

    for r in range(runs):
        X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=siglen, do_print=init)
        init=False

        print(f'Simulation #{r}')

        model = CesnModel_V2(r1_nnodes=rsize, r2_nnodes=rsize)
        model.train_r1(input=X_train, target=Y_train, reset=reset_state)
        model.train_r2(input=X_train, target=Y_train, reset=reset_state)

        results = model.test(input=X_test, target=Y_test, condition=c, input_sigma=noise, reset=reset_state)
        joint, upper, lower, avg = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test)

        acc_joint.append(joint)
        acc_upper.append(upper)
        acc_lower.append(lower)
        acc_avg.append(avg)

        # Save

#        if save_data==True:
#            signal_ids = [i for i, signal in enumerate(Y_test) for _ in range(len(signal))]
#
#            sig_df = pd.DataFrame(signal_ids, columns=['signal'])
#            sig_df.to_csv(f'data/v2/signal_ids.csv', index=False)
#
#            ytest = np.vstack(Y_test)
#            ytest_df = pd.DataFrame(ytest, columns=[f'X{i}' for i in range(len(ytest[0]))])
#            ytest_df.to_csv(f'data/v2/ytest.csv', index=False)
#
#            y1 = np.vstack(results[0])
#            y2 = np.vstack(results[1])
#            # r1 = np.vstack(results[2])
#            # r2 = np.vstack(results[3])
#
#            for (name, item) in zip(['y1', 'y2'], [y1, y2]): # r1, r2
#                pred_df = pd.DataFrame(item, columns=[f'X{i}' for i in range(len(item[0]))])
#                pred_df.to_csv(f'data/v2/{name}_{c}_r={rsize}_reset={reset_state}_noise={noise}_siglen={siglen}_sim={r}.csv', index=False)

    store_coupling.append(c)
    store_accuracies_joint.append(acc_joint)
    store_accuracies_upper.append(acc_upper)
    store_accuracies_lower.append(acc_lower)
    store_accuracies_avg.append(acc_avg)
    
    print(f'{c}; nnodes: {rsize};  N: {runs}; pcorrect (joint): {np.mean(acc_joint)}')

if save_data==True:
    conditions_repeated = np.repeat(store_coupling, runs)

    joint_flat = np.array(store_accuracies_joint).flatten()
    upper_flat = np.array(store_accuracies_upper).flatten()
    lower_flat = np.array(store_accuracies_lower).flatten()
    avg_flat = np.array(store_accuracies_avg).flatten()

    acc_df = pd.DataFrame({
        'condition': conditions_repeated,
        'joint': joint_flat,
        'upper': upper_flat,
        'lower': lower_flat,
        'avg': avg_flat
    })

    acc_df.to_csv(f'data/v2/accuracies_r={rsize}_reset={reset_state}_noise={noise}_siglen={siglen}.csv', index=False)

# Plot (full barplot)

# x_categories = ['independent', 'interaction', 'integration']
# x_indices = np.arange(len(x_categories))
# 
# datasets = [np.array(store_accuracies_joint), 
#             np.array(store_accuracies_upper), 
#             np.array(store_accuracies_lower),
#             np.array(store_accuracies_avg)]
# labels = [r'$ESN_{dyad}$', r'$ESN_{max}$', r'$ESN_{min}$', r'$ESN_{avg}$']
# 
# bar_width = 0.2
# offsets = np.linspace(-bar_width*1.5, bar_width*1.5, len(datasets))
# 
# for y_values, label, offset in zip(datasets, labels, offsets):
#     if y_values.shape[1] == 1:
#         y_means = y_values.flatten()
#         plt.bar(x_indices + offset, y_means, width=bar_width, label=label)
#     else:
#         y_means = np.mean(y_values, axis=1)
#         y_sems = np.std(y_values, axis=1, ddof=1) / np.sqrt(y_values.shape[1])
#         plt.bar(x_indices + offset, y_means, width=bar_width, yerr=y_sems,
#                 capsize=5, label=label)
#         
#     # add mean value above each bar
#     for x, y in zip(x_indices + offset, y_means):
#         plt.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=7)
# 
# plt.xticks(x_indices, x_categories)
# # plt.xlabel('Condition')
# plt.ylabel('avg. pcorrect')
# plt.ylim(0, 1)
# plt.grid(axis='y', alpha=0.7)
# plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
# plt.tight_layout()
# # plt.savefig(f'plt_figs/v2/barplot_full_performance_runs={runs}_rsize={rsize}_reset={reset_state}_noise={noise}_siglen={siglen}.png', dpi=300)
# plt.show()



# Plot (partial barplot)

# # independent measure(s)
# woc = store_accuracies_joint[0]
# upper = store_accuracies_upper[0]
# lower = store_accuracies_lower[0]
# avg = store_accuracies_avg[0]
# 
# # interaction measure(s)
# joint = store_accuracies_joint[1]
# 
# # collect data
# datasets = [lower, upper, avg, woc, joint]
# labels   = [r"$ESN_{min}$", r"$ESN_{max}$", r"$ESN_{avg}$", r"$ESN_{woc}$", r"$ESN_{dyad}$"]
# 
# means = [np.mean(arr) for arr in datasets]
# sems  = [np.std(arr, ddof=1) / np.sqrt(len(arr)) for arr in datasets]
# 
# x_indices = np.arange(len(datasets))
# 
# plt.figure(figsize=(7,4))
# bars = plt.bar(x_indices, means, yerr=sems, capsize=5, width=0.6, color="lightgray", edgecolor="black")
# plt.xticks(x_indices, labels)
# plt.ylabel("Avg. Pcorrect")
# plt.ylim(0, 1)
# plt.grid(axis='y', alpha=0.6)
# 
# # annotate
# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width()/2,
#         height + 0.02,
#         f"{means[i]:.2f}",
#         ha='center', va='bottom'
#     )
# 
# plt.tight_layout()
# # plt.savefig(f'plt_figs/v2/barplot_performance_runs={runs}_rsize={rsize}_reset={reset_state}_noise={noise}_siglen={siglen}.png', dpi=300)
# plt.show()
