import time
from datetime import timedelta, datetime
import pandas as pd
import reservoirpy as rpy
from data_processing import *
from cesn_model import *


rpy.verbosity(0)
global_seed = 42
rng = np.random.default_rng(global_seed)

base_path = f"data/predictions"
runs = 20


### RUN ###

results_list = []
preds_list = []

start_time = time.time()

for sim in range(runs):
    np.random.seed(rng.integers(0, 2**32-1))
    r_seeds = rng.integers(0, 2**32-1, size=2).tolist()

    # sample data
    X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True)

    # train networks
    model = CesnModel_Multi(ensemble_size=2, nnodes=800, in_plink=0.1, rc_plink=0.1, seed=r_seeds, do_print=False)
    model.train(inputs=X_train, targets=Y_train, teacherfb_sigma=0.2)

    # test networks
    for cond in ["autocentric", "polycentric"]:     
              
        results = model.test(inputs=X_test, targets=Y_test, condition=cond, input_sigma=2.0)
        # joint, upper, lower, avg = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test)

        # results_list.append({
        #     "run": sim,
        #     "condition": cond,
        #     "joint": joint,
        #     "upper": upper,
        #     "lower": lower,
        #     "avg": avg
        # })

        y1 = np.vstack(results[0]) ## FIX ##
        y2 = np.vstack(results[1])

        for (name, item) in zip(['y1', 'y2'], [y1, y2]):
            df = pd.DataFrame(item, columns=[f'X{i}' for i in range(len(item[0]))])
            df.to_csv(base_path + f'_{name}_sim={sim}_cond={cond}.csv', index=False)

    # save
    if sim==0:
        signal_ids = [i for i, signal in enumerate(Y_test) for _ in range(len(signal))]
        sig_df = pd.DataFrame(signal_ids, columns=['signal'])
        sig_df.to_csv(base_path + f'_signal_ids.csv', index=False)

        ytest = np.vstack(Y_test)
        ytest_df = pd.DataFrame(ytest, columns=[f'X{i}' for i in range(len(ytest[0]))])
        ytest_df.to_csv(base_path + f'_ytest.csv', index=False)

# # log pcorrect
# df = pd.DataFrame(results_list)
# means = df[["joint", "upper", "lower", "avg"]].mean()
# print(f"Average over {runs} runs:")
# print(means)

print(f"Overall execution time: {str(timedelta(seconds=time.time() - start_time))}")