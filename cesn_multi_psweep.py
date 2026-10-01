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
    return degree / nnodes


def check_isolation(tag, log_path, rng): # confirm independence of parallel processes
    with open(log_path, "a") as f:
        f.write(f"[{tag}] PID={os.getpid()} starting isolation test...\n")
        val = float(rng.random())
        f.write(f"[{tag}] PID={os.getpid()} random_val={val:.6f} \n")


def run_simulations(runs, esize, rsize, plink, tsigma, rconn, rng): # runs N simulations given (r,p,t,c)
    results_list = []

    for _ in range(runs):
        np.random.seed(rng.integers(0, 2**32-1))
        r_seeds = rng.integers(0, 2**32-1, size=esize).tolist()

        X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, zscore=True, do_print=False)

        model = CesnModel_Multi(ensemble_size=esize, nnodes=rsize, in_plink=plink, rc_plink=rconn, seed=r_seeds, do_print=False)
        model.train(inputs=X_train, targets=Y_train, teacherfb_sigma=tsigma)
        outputs = model.test(inputs=X_test, targets=Y_test, condition="polycentric", input_sigma=2.0) # input_sigma=2.0
        results = model.accuracy(predictions=outputs, targets=Y_test)

        results_list.append({"results": results, "outputs": outputs, "targets": np.asarray(Y_test)})

    return results_list


def run_batch(batch_idx, batch, runs, base_path, global_seed, main_log_path): # runs batch of (e,r,p,t,c) values
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

        # collect data
        batch_csv_path = f"{base_path}_batch_{batch_idx}"
        esn_csv = f"{batch_csv_path}.csv"
        net_csv = f"{batch_csv_path}_network.csv"

        esn_rows = []
        net_rows = []

        for esize, rsize, plink, tsigma in batch:
            # ensure integers
            esize = int(esize)
            rsize = int(rsize)

            # calculate sparsity to ensure out_degree=10
            rconn = reservoir_sparsity(rsize)

            # run simulations per parameterization
            simulations = run_simulations(runs, esize, rsize, plink, tsigma, rconn, rng)

            # compute summary statistics
            joint_accs = np.array([s["results"]["joint_acc"] for s in simulations])
            # indiv_accs = np.array([s["indiv_accs"] for s in simulations])

            mean_joint = joint_accs.mean()
            se_joint = joint_accs.std(ddof=1) / np.sqrt(len(joint_accs))

            # mean_indiv = indiv_accs.mean(axis=0)
            # se_indiv = indiv_accs.std(axis=0, ddof=1) / np.sqrt(indiv_accs.shape[0])

            # log values
            print(
                f"esize={esize} | "
                f"rsize={rsize} | "
                f"plink={plink:.3f} | "
                f"tsigma={tsigma:.3f} | "
                f"rconn={rconn:.3f} | "
                f"joint={mean_joint:.4f} " ,
                # f"indiv={np.round(mean_indiv, 4)}",
                flush=True
            )

            # store data
            esn_rows.append({
                "esize": esize,
                "rsize": rsize,
                "plink": plink,
                "tsigma": tsigma,
                "rconn": rconn,
                "acc": mean_joint,
                "se": se_joint
            })

#             for sim_idx, sim in enumerate(simulations):
#                 outputs = sim["outputs"]      # (signal, esize, timestep, readout)
#                 targets = sim["targets"]
# 
#                 n_signals, n_networks, n_timesteps, n_readouts = outputs.shape
# 
#                 for signal_idx in range(n_signals):
#                     target = np.argmax(targets[signal_idx, 0])
# 
#                     for net_idx in range(n_networks):
#                         for t in range(n_timesteps):
#                             row = {
#                                 "esize": esize,
#                                 "rsize": rsize,
#                                 "plink": plink,
#                                 "tsigma": tsigma,
#                                 "rconn": rconn,
#                                 "simulation": sim_idx + 1,
#                                 "signal": signal_idx + 1,
#                                 "network": net_idx + 1,
#                                 "timestep": t + 1,
#                                 "target": target + 1
#                             }
# 
#                             # network output at this timestep
#                             net_out = outputs[signal_idx, net_idx, t]
#                             row.update({f"RO{i+1}": net_out[i] for i in range(n_readouts)})
#                             net_rows.append(row)

#             # save output data
#             net_df = pd.DataFrame(net_rows)
#             net_df.to_csv(net_csv, index=False)

        # save performance data
        esn_df = pd.DataFrame(esn_rows)
        esn_df.to_csv(esn_csv, index=False)

        # print(f"Results saved to {batch_csv_path}", flush=True)

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
    runs = 10 # simulations per parameterization
    base_path = f"data/v3/temp/psweep" # readouts_n=2_autocentric/psweep"

    # redirect stdout and stderr to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = f"{base_path}_log_{timestamp}.txt"

    main_log = open(main_log_path, "w", buffering=1)
    sys.stdout = main_log
    sys.stderr = main_log

    # hyperparams
    esize_range =   [4]                                # ensemble size
    rsize_range =   [2560]       # reservoir size
    plink_range =   [0.1]                                       # input/fb connectivity
    tsigma_range =  [0.2, 0.4, 0.8, 1.6, 3.2, 6.4]              # noise added to teacher forcing
    # rconn_range =   []                                        # reservoir internal connectivity

    print(f"Global seed: {global_seed}")
    print(f"{runs} simulations per parameterization")

    # create parameter combinations
    T, E, R, P = np.meshgrid(tsigma_range, esize_range, rsize_range, plink_range, indexing='ij')      # rconn_range
    param_combs = np.column_stack([E.ravel(), R.ravel(), P.ravel(), T.ravel()])                       # C.ravel()

    # batch parameters for sweep
    batch_size = 1
    total = param_combs.shape[0]
    batches = [param_combs[i:i+batch_size] for i in range(0, total, batch_size)]

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
