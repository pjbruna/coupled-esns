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

def check_isolation(tag, log_path): # confirm independence of parallel processes
    with open(log_path, "a") as f:
        f.write(f"[{tag}] PID={os.getpid()} starting isolation test...\n")

        rand_py = random.random()
        rand_np = np.random.rand()
        dummy = np.random.randn(3, 3)
        obj_hash = hash(dummy.tobytes())

        f.write(f"[{tag}] PID={os.getpid()} random_py={rand_py:.6f} "
                f"random_np={rand_np:.6f} hash={obj_hash}\n")


def run_simulations(runs, rsize, plink, tsigma): # runs N simulations given (r,p,t)
    rsize = int(rsize)
    results_list = []

    for _ in range(runs):
        X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, do_print=False)

        model = CesnModel_V2(nnodes=[rsize,rsize], in_plink=[plink,plink])
        model.train_r1(input=X_train, target=Y_train, teacherfb_sigma=tsigma)
        model.train_r2(input=X_train, target=Y_train, teacherfb_sigma=tsigma)

        for cond in ["auto", "allo", "poly_parall", "poly_integr"]:   
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


def run_batch(batch_idx, batch, runs, base_path, seed, main_log_path): # runs batch of (r,p,t) values
    rpy.verbosity(0)
    np.random.seed(seed + batch_idx)
    random.seed(seed + batch_idx)
    rpy.set_seed(seed + batch_idx)
    check_isolation(f"batch_{batch_idx}", main_log_path)

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
        for rsize, plink, tsigma in batch:
            results = run_simulations(runs, rsize, plink, tsigma)

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
            summary["rsize"] = rsize
            summary["plink"] = plink
            summary["tsigma"] = tsigma

            # log outcomes
            accuracies_list.extend(summary.to_dict(orient="records"))

            # print
            print_m = np.round(summary.loc[(summary['condition']=='auto') & (summary['measure']=='avg'), 'm'].item(),4)
            print_se = np.round(summary.loc[(summary['condition']=='auto') & (summary['measure']=='avg'), 'se'].item(),4)
            print(f"rsize: {rsize:<6} plink: {plink:<5} tsigma: {tsigma:<5} pcorrect: {print_m:<5} se: {print_se:<5}")


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
    seed = 42
    base_path = f"data/v2_psweep/psweep"

    # redirect stdout and stderr to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = f"{base_path}_log_{timestamp}.txt"

    main_log = open(main_log_path, "w", buffering=1)
    sys.stdout = main_log
    sys.stderr = main_log

    # hyperparams
    rsize_range = [50, 100, 200, 400, 800, 1600]      # reservoir size
    plink_range = [0.1, 0.55, 1.0]                    # input/fb connectivity
    tsigma_range = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]     # noise added to teacher forcing
    runs = 10                                         # simulations per parameterization

    print(f"Random seed: {seed}")
    print(f"{runs} simulations per (r,p,t)")

    # batch parameters for sweep
    R, P, T = np.meshgrid(rsize_range, plink_range, tsigma_range, indexing='ij')
    param_combs = np.column_stack([R.ravel(), P.ravel(), T.ravel()])

    batch_size = 18
    total = param_combs.shape[0]
    batches = [param_combs[i:i+batch_size] for i in range(0, total, batch_size)]

    # start
    print(f"Launching {len(batches)} batches in parallel...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max(1,os.cpu_count()-1)) as executor:
        futures = {executor.submit(run_batch, i+1, batch, runs, base_path, seed, main_log_path): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            bidx, logpath, csvpath = future.result()
            print(f"Completed batch {bidx}, log: {logpath}, csv: {csvpath}", flush=True)


    # log total runtime
    elapsed = time.time() - start_time
    print(f"Overall execution time: {str(timedelta(seconds=elapsed))}")
    main_log.close()
