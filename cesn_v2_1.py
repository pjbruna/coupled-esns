import reservoirpy as rpy
from data_processing import *
from cesn_model import *

rpy.verbosity(0)
seed = 42


# Hyperparams
rsize = 500 # reservoir size
reset_state = 'zero' # reset reservoirs between signals
noise_range = [0, 0.1, 0.2, 0.3, 0.4] # noise added to input signals during testing (equal vs unequal conditions)
runs = 50


# Run model

base_path = f"data/v2_1/accuracies_r={rsize}_reset={reset_state}"
results_list = []

for sim in range(runs):
    X_train, Y_train, X_test, Y_test = generate_jvowels(signal_length=10, do_print=(sim==0))

    model = CesnModel_V2(r1_nnodes=rsize, r2_nnodes=rsize)
    model.train_r1(input=X_train, target=Y_train, reset=reset_state)
    model.train_r2(input=X_train, target=Y_train, reset=reset_state)

    for i, noise_1 in enumerate(noise_range):
        for noise_2 in noise_range[i:]:
            for cond in ["auto", "allo", "poly_parall", "poly_integr"]:   
                print(f"Simulation #{sim} ...{cond:<15} | noise: {noise_1:.1f} // {noise_2:.1f}")  
                      
                results = model.test(input=X_test, target=Y_test, condition=cond, input_sigma=[noise_1, noise_2], reset=reset_state)
                joint, upper, lower, avg = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test)

                results_list.append({
                    "run": sim,
                    "condition": cond,
                    "joint": joint,
                    "upper": upper,
                    "lower": lower,
                    "avg": avg,
                    "noise_1": noise_1,
                    "noise_2": noise_2
                })

    if (sim + 1) % 10 == 0:
        save_progress(results_list, base_path + "_partial.csv")

# final save
save_progress(results_list, base_path + "_final.csv", final=True)
