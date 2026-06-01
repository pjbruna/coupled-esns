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

def reservoir_sparsity(nnodes, degree=10): # calculate reservoir sparsity based on number of reservoir nodes
    return [degree / n for n in nnodes]


def check_isolation(tag, log_path, rng): # confirm independence of parallel processes
    with open(log_path, "a") as f:
        f.write(f"[{tag}] PID={os.getpid()} starting isolation test...\n")
        val = float(rng.random())
        f.write(f"[{tag}] PID={os.getpid()} random_val={val:.6f} \n")


def run_simulations(runs, esize, rsize, plink, tsigma, rconn, rng): # runs N simulations given (r,p,t,c)
    rsize = [int(r) for r in rsize]
    results_list = []

    for _ in range(runs):
        np.random.seed(rng.integers(0, 2**32-1))
        r_seeds = rng.integers(0, 2**32-1, size=esize).tolist()

        X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True, do_print=False)

        model = CesnModel_Multi(ensemble_size=esize, nnodes=rsize, in_plink=plink, rc_plink=rconn, seed=r_seeds, print=False)
        model.train(inputs=X_train, targets=Y_train, teacherfb_sigma=tsigma)
        outputs = model.test(inputs=X_test, targets=Y_test, condition="polycentric", input_sigma=0)
        results = model.accuracy(predictions=outputs, targets=Y_test)

        results_list.append(results)

    return results_list


def run_batch(batch_idx, batch, runs, esize, base_path, global_seed, main_log_path): # runs batch of (e,r,p,t,c) values
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

        ens_rows = []
        net_rows = []
        sim_idx = 0

        for rsize, plink, tsigma in batch:
            rconn = reservoir_sparsity(rsize)
            simulations = run_simulations(runs, esize, rsize, plink, tsigma, rconn, rng)

            # compute summary statistics

            joint_accs = np.array([s["joint_acc"] for s in simulations])
            indiv_accs = np.array([s["indiv_accs"] for s in simulations])

            mean_joint = joint_accs.mean()
            se_joint = joint_accs.std(ddof=1) / np.sqrt(len(joint_accs))

            mean_indiv = indiv_accs.mean(axis=0)
            # se_indiv = indiv_accs.std(axis=0, ddof=1) / np.sqrt(indiv_accs.shape[0])

            # log values (assumes homogenous ensembles)
            print(
                f"rsize={rsize[0]} | "
                f"plink={plink[0]:.3f} | "
                f"tsigma={tsigma[0]:.3f} | "
                f"rconn={rconn[0]:.3f} | "
                f"joint={mean_joint:.4f} | "
                f"indiv={np.round(mean_indiv, 4)}",
                flush=True
            )

            # store data
            ens_rows.append({
                "sim": sim_idx,
                "batch": batch_idx,
                "esize": esize,
                "rsize": rsize[0],          # assumes homogenous ensemble
                "plink": plink[0],          # assumes homogenous ensemble
                "tsigma": tsigma[0],        # assumes homogenous ensemble
                "rconn": rconn[0],          # assumes homogenous ensemble
                "acc": mean_joint,
                "se": se_joint
            })

            # for net_idx in range(esize):  # if ensembles are heterogeneous...
            #     net_rows.append({
            #         "sim": sim_idx,
            #         "batch": batch_idx,
            #         "net": net_idx+1,
            #         "rsize": rsize[net_idx],
            #         "plink": plink[net_idx],
            #         "tsigma": tsigma[net_idx],
            #         "rconn": rconn[net_idx],
            #         "acc": mean_indiv[net_idx],
            #         "se": se_indiv[net_idx]
            #     })

            sim_idx += 1

        # save batch
        ens_df = pd.DataFrame(ens_rows)
        # net_df = pd.DataFrame(net_rows)

        batch_csv_path = f"{base_path}_batch_{batch_idx}"
        ens_df.to_csv(f"{batch_csv_path}.csv", index=False)
        # net_df.to_csv(f"{batch_csv_path}_network.csv", index=False)

        print(f"Results saved to {batch_csv_path}", flush=True)

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
    runs = 10 # simulations per parameterization
    esize = 16

    base_path = f"data/v3/matched_budget/ensemble_{esize}/psweep_{analysis}"

    # redirect stdout and stderr to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = f"{base_path}_log_{timestamp}.txt"

    main_log = open(main_log_path, "w", buffering=1)
    sys.stdout = main_log
    sys.stderr = main_log

    # hyperparams
    rsize_range =   [400/esize, 800/esize, 1600/esize]        # reservoir size
    plink_range =   [0.1] # [0.1, 0.3, 0.5]                   # input/fb connectivity
    tsigma_range =  [0.2, 0.4, 0.8, 1.6, 3.2, 6.4]            # noise added to teacher forcing
    # rconn_range =   []                                        # reservoir internal connectivity

    print(f"Global seed: {global_seed}")
    print(f"{runs} simulations per parameterization")

    # create parameter combinations
    R, P, T = np.meshgrid(rsize_range, plink_range, tsigma_range, indexing='ij')      # rconn_range
    param_combs = np.column_stack([R.ravel(), P.ravel(), T.ravel()])                  # C.ravel()

    # create full parameter sweeps
    if analysis=="mixed":

        # shuffle for mixed-heads analysis
        mixed_list = []
        for e in range(esize):
            shuffled_param_combs = param_combs.copy()
            np.random.shuffle(shuffled_param_combs)
            mixed_list.append(shuffled_param_combs)

        combs_sweep = np.stack(mixed_list, axis=2)

    else:
        combs_sweep = np.stack([param_combs] * esize, axis=2)

    # batch parameters for sweep
    batch_size = 6 # 18
    total = combs_sweep.shape[0]
    batches = [combs_sweep[i:i+batch_size] for i in range(0, total, batch_size)]

    # start
    print(f"Launching {len(batches)} batches in parallel...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max(1,os.cpu_count()-1)) as executor:
        futures = {executor.submit(run_batch, i+1, batch, runs, esize, base_path, global_seed, main_log_path): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            bidx, logpath, csvpath = future.result()
            print(f"Completed batch {bidx}, log: {logpath}, csv: {csvpath}", flush=True)

    # log total runtime
    elapsed = time.time() - start_time
    print(f"Overall execution time: {str(timedelta(seconds=elapsed))}")
    main_log.close()
