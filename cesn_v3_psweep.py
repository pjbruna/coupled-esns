import os
import sys
import time
from datetime import timedelta, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import numpy as np
import pandas as pd
import reservoirpy as rpy
from data_processing import *
from cesn_model import *


# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def check_isolation(tag, log_path, rng): # confirm independence of parallel processes
    with open(log_path, "a") as f:
        f.write(f"[{tag}] PID={os.getpid()} starting isolation test...\n")
        val = float(rng.random())
        f.write(f"[{tag}] PID={os.getpid()} random_val={val:.6f} \n")


def run_simulations(runs, rsize, plink, tsigma, rconn, rng): # runs N simulations given (r,p,t)
    rsize = [int(r) for r in rsize]
    results_list = []

    for _ in range(runs):
        np.random.seed(rng.integers(0, 2**32-1))
        r_seeds = rng.integers(0, 2**32-1, size=2).tolist()

        X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True, do_print=False)

        model = CesnModel_V3(nnodes=rsize, in_plink=plink, rc_plink=rconn, seed=r_seeds)
        model.train_r1(input=X_train, target=Y_train, teacherfb_sigma=tsigma[0])
        model.train_r2(input=X_train, target=Y_train, teacherfb_sigma=tsigma[1])

        for cond in ["auto", "allo", "poly"]:   
            results = model.test(input=X_test, target=Y_test, condition=cond)
            joint, upper, lower, avg = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test)

            results_list.append({
                "condition": cond,
                "joint": joint,
                "upper": upper,
                "lower": lower,
                "avg": avg,
            })

    return results_list


def run_batch(batch_idx, batch, runs, base_path, global_seed, main_log_path): # runs batch of (r,p,t) values
    rpy.verbosity(0)
    rng = np.random.default_rng(global_seed + batch_idx)
    check_isolation(f"batch_{batch_idx}", main_log_path, rng)

    batch_log_path = f"{base_path}_batch_{batch_idx}.log"
    bstart = time.time()

    with open(batch_log_path, "w", buffering=1) as blog:
        sys.stdout = blog
        sys.stderr = blog

        print("=" * 45)
        print(f"Batch {batch_idx} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Contains {len(batch)} parameter combinations")
        print("-" * 45)

        accuracies_list = []
        for rsize, plink, tsigma, rconn in batch:
            results = run_simulations(runs, rsize, plink, tsigma, rconn, rng)

            # calculate mean and standard error per condition/measure
            tempdf = pd.DataFrame(results)
            tempdf_long = tempdf.melt(id_vars="condition", 
                                      value_vars=["joint", "upper", "lower", "avg"], 
                                      var_name="measure", 
                                      value_name="value")

            summary = (
                tempdf_long
                .groupby(["condition", "measure"], as_index=False)
                .agg(
                    m=("value", "mean"),
                    se=("value", lambda x: x.std(ddof=1) / np.sqrt(len(x)))
                )
            )

            # add parameter values to summary
            summary["rsize_1"] = rsize[0]
            summary["plink_1"] = plink[0]
            summary["tsigma_1"] = tsigma[0]
            summary["rconn_1"] = rconn[0]
            summary["rsize_2"] = rsize[1]
            summary["plink_2"] = plink[1]
            summary["tsigma_2"] = tsigma[1]
            summary["rconn_2"] = rconn[1]

            # log outcomes
            accuracies_list.extend(summary.to_dict(orient="records"))

            # print
            print_m = np.round(summary.loc[(summary['condition']=='auto') & (summary['measure']=='avg'), 'm'].item(),4)
            print_se = np.round(summary.loc[(summary['condition']=='auto') & (summary['measure']=='avg'), 'se'].item(),4)
            print(f"rsize: ({rsize[0]:<6}, {rsize[1]:<6}) | "
                  f"plink: ({plink[0]:<5}, {plink[1]:<5}) | "
                  f"tsigma: ({tsigma[0]:<5}, {tsigma[1]:<5}) | "
                  f"rconn: ({rconn[0]:<5}, {rconn[1]:<5}) | "
                  f"pcorrect: {print_m:<5} | "
                  f"se: {print_se:<5}")

        # save batch
        batchdf = pd.DataFrame(accuracies_list)
        batch_csv_path = f"{base_path}_batch_{batch_idx}.csv"
        batchdf.to_csv(batch_csv_path, index=False)

        print(f"Results saved to {batch_csv_path} ({len(batchdf)} entries).", flush=True)

        # log batch end time
        belapsed = time.time() - bstart
        print(f"Batch {batch_idx} execution time: {str(timedelta(seconds=belapsed))}", flush=True)

        return batch_idx, batch_log_path, batch_csv_path


# ---------------------------------------------------------
# RUN BATCHES IN PARALLEL
# ---------------------------------------------------------

if __name__ == "__main__":
    # setup
    global_seed = 42
    np.random.seed(global_seed)
    analysis="same" # same-heads analysis or mixed-heads analysis
    base_path = f"data/v3/psweep_{analysis}"
    runs = 10 # simulations per parameterization

    # redirect stdout and stderr to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = f"{base_path}_log_{timestamp}.txt"

    main_log = open(main_log_path, "w", buffering=1)
    sys.stdout = main_log
    sys.stderr = main_log

    # hyperparams
    rsize_range =   [50, 100, 200, 400, 800, 1600]      # reservoir size
    plink_range =   [0.1, 0.3, 0.5]                     # input/fb connectivity
    tsigma_range =  [0.2, 0.4, 0.8, 1.6, 3.2, 6.4]      # noise added to teacher forcing
    rconn_range =   [0.1]                               # reservoir internal connectivity

    print(f"Global seed: {global_seed}")
    print(f"{runs} simulations per (r,p,t,c)")

    # create parameter combinations
    R, P, T, C = np.meshgrid(rsize_range, plink_range, tsigma_range, rconn_range, indexing='ij')
    param_combs = np.column_stack([R.ravel(), P.ravel(), T.ravel(), C.ravel()])

    # shuffle for mixed-heads analysis
    shuffled_param_combs = param_combs.copy()
    np.random.shuffle(shuffled_param_combs)

    # create full parameter sweeps
    combs_same = np.stack([param_combs, param_combs], axis=2)
    combs_mixed = np.stack([param_combs, shuffled_param_combs], axis=2)

    # batch parameters for sweep
    batch_size = 18
    if analysis=="same":
        total = combs_same.shape[0]
        batches = [combs_same[i:i+batch_size] for i in range(0, total, batch_size)]
    if analysis=="mixed":
        total = combs_mixed.shape[0]
        batches = [combs_mixed[i:i+batch_size] for i in range(0, total, batch_size)]

    # start
    print(f"Launching {len(batches)} batches in parallel...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max(1,os.cpu_count()-1)) as executor:
        futures = {executor.submit(run_batch, i+1, batch, runs, base_path, global_seed, main_log_path): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            bidx, logpath, csvpath = future.result()
            print(f"Completed batch {bidx}, log: {logpath}, csv: {csvpath}", flush=True)

    # log total runtime
    elapsed = time.time() - start_time
    print(f"Overall execution time: {str(timedelta(seconds=elapsed))}")
    main_log.close()
