import time
from datetime import timedelta
import reservoirpy as rpy
from data_processing import *
from cesn_model import *


rpy.verbosity(0)
global_seed = 42
rng = np.random.default_rng(global_seed)


### HYPERPARAMS ###

rsize_range = [800, 1600]    # reservoir size
plink_range = [0.1]          # input/fb connectivity
tsigma_range = [0.2, 0.4]    # noise added to teacher forcing

reset_state = 'zero'         # reset reservoirs between signals
noise_range = [0.5, 2.0]     # noise added to input signals during testing (equal vs unequal conditions)
runs = 20                    # dyad simulations


### RUN ###

base_path = f"data/v3/speed_acc/2026_06_14/"
results_list = []

start_time = time.time()

for sim in range(runs):
    np.random.seed(rng.integers(0, 2**32-1))
    r_seeds = rng.integers(0, 2**32-1, size=2).tolist()

    # sample data
    X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True)

    # sample heterogenous networks
    rsize = select_param(rsize_range, 2, rng)
    plink = select_param(plink_range, 2, rng)
    tsigma = select_param(tsigma_range, 2, rng)

    print(
        f"{f'Simulation #{sim}':<20}"
        f"{f'ESN_1: ({rsize[0]:>4}, {plink[0]:>6.2f}, {tsigma[0]:>7.1f})':<40}"
        f"{f'ESN_2: ({rsize[1]:>4}, {plink[1]:>6.2f}, {tsigma[1]:>7.1f})':<40}"
    )

    # train networks
    model = CesnModel_V3(nnodes=rsize, in_plink=plink, seed=r_seeds)
    model.train_r1(input=X_train, target=Y_train, teacherfb_sigma=tsigma[0])
    model.train_r2(input=X_train, target=Y_train, teacherfb_sigma=tsigma[1])

    # test networks
    for i, noise_1 in enumerate(noise_range):
        for noise_2 in noise_range[i:]:
            for cond in ["auto", "poly"]:     
                      
                results = model.test(input=X_test, target=Y_test, condition=cond, input_sigma=[noise_1, noise_2], reset=reset_state)
                joint, upper, lower, avg = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test)

                y1 = np.vstack(results[0])
                y2 = np.vstack(results[1])

                for (name, item) in zip(['y1', 'y2'], [y1, y2]):
                    df = pd.DataFrame(item, columns=[f'X{i}' for i in range(len(item[0]))])
                    df.to_csv(base_path + f'_{name}_sim={sim}_noise={noise_1}_{noise_2}_cond={cond}.csv', index=False)

                results_list.append({
                    "run": sim,
                    "condition": cond,
                    "joint": joint,
                    "upper": upper,
                    "lower": lower,
                    "avg": avg,
                    "noise_1": noise_1,
                    "noise_2": noise_2,
                    "rsize_1": rsize[0],
                    "rsize_2": rsize[1],
                    "tsigma_1": tsigma[0],
                    "tsigma_2": tsigma[1]

                })

    if (sim + 1) % 10 == 0:
        save_progress(results_list, base_path + f"accuracies_N={runs}_bxbr_partial.csv")

# final save
save_progress(results_list, base_path + f"accuracies_N={runs}_bxbr_final.csv", final=True)

signal_ids = [i for i, signal in enumerate(Y_test) for _ in range(len(signal))]
sig_df = pd.DataFrame(signal_ids, columns=['signal'])
sig_df.to_csv(base_path + f'_signal_ids.csv', index=False)

ytest = np.vstack(Y_test)
ytest_df = pd.DataFrame(ytest, columns=[f'X{i}' for i in range(len(ytest[0]))])
ytest_df.to_csv(base_path + f'_ytest.csv', index=False)

print(f"Overall execution time: {str(timedelta(seconds=time.time() - start_time))}")
