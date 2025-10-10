import sys
import time
from datetime import timedelta, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import reservoirpy as rpy
from data_processing import *
from cesn_model import *


rpy.verbosity(0)
seed = 42
base_path = f"data/v2_1/paramsweep"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"{base_path}_log_{timestamp}.txt"


# Redirect stdout and stderr to a file
with open(log_path, "w", buffering=1) as f:
    sys.stdout = f
    sys.stderr = f


    # Hyperparams
    rsize_range = np.round(np.linspace(20, 1500, 10), 0)      # reservoir size
    plink_range = np.round(np.linspace(0.1, 1.0, 10), 2)     # input/fb connectivity
    tsigma_range = np.round(np.linspace(0.0, 1.0, 10), 2)    # noise added to teacher forcing
    runs = 25                                               # simulations per parameterization

    print(f"Random seed: {seed}")
    print(f"{runs} simulations per (r,p,t)")


    # Log start time
    start_time = time.time()


    # Batch parameters for sweep
    R, P, T = np.meshgrid(rsize_range, plink_range, tsigma_range, indexing='ij')
    param_combs = np.column_stack([R.ravel(), P.ravel(), T.ravel()])

    batch_size = 25
    total = param_combs.shape[0]
    batches = [param_combs[i:i+batch_size] for i in range(0, total, batch_size)]


    # Define model

    def run_simulations(runs, rsize, plink, tsigma):
        rsize = int(rsize)
        results_list = []

        for sim in range(runs):
            # print(f"Simulation #{sim}...")

            X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, do_print=False)

            model = CesnModel_V2(nnodes=[rsize,rsize], plink=[plink,plink])
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


    # Run model

    for batch_idx, batch in enumerate(batches, 1):
        bstart = time.time()

        print("="*45)
        print(f"Batch {batch_idx}/{len(batches)} has {len(batch)} combinations")
        print("-"*45)
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
            print_m = np.round(summary.loc[(summary['condition']=='auto') & (summary['measure']=='avg'), 'm'].item(),2)
            print_se = np.round(summary.loc[(summary['condition']=='auto') & (summary['measure']=='avg'), 'se'].item(),2)
            print(f"rsize: {rsize:<6} plink: {plink:<5} tsigma: {tsigma:<5} pcorrect: {print_m:<5} se: {print_se:<5}")


        # save batch
        batchdf = pd.DataFrame(accuracies_list)
        batch_path = f"{base_path}_batch_{batch_idx}.csv"
        batchdf.to_csv(batch_path, index=False)

        print(f"Results saved to {batch_path} ({len(batchdf)} entries).")

        # Log batch end time
        bend = time.time()
        belapsed = bend - bstart
        bformatted = str(timedelta(seconds=belapsed))
        print(f"Batch {batch_idx}/{len(batches)} execution time: {bformatted}")

    # Log end time
    end_time = time.time()
    elapsed = end_time - start_time
    formatted = str(timedelta(seconds=elapsed))
    print(f"Overall execution time: {formatted}")
