import numpy as np
import random
import math
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *
from functions import *


rpy.verbosity(0)

seed = 42
random.seed(seed)


# Redirect stdout and stderr to a file
with open("anneal_log_CONT.txt", "w", buffering=1) as f:
    sys.stdout = f
    sys.stderr = f

    # Print seed
    print(f"Random seed: {seed}")

    # Log start time
    start_time = time.time()

    # Define model

    def run_model(runs=None, coupling_num=None, r1_size=None, r2_size=None, train_sample=None):

        # Training data

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

        # Model

        store_accuracies = []

        for c in [0.0, 0.5]: # np.linspace(0.0, 1.0, num=coupling_num):
            # c = round(c, 1)
            acc = []

            for r in range(runs):
                print(f'Run #{r}')
                model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=c)

                model.train_r1(input=X_train1, target=Y_train1)
                model.train_r2(input=X_train2, target=Y_train2)

                results = model.test(input=X_test, target=Y_test, noise_scale=0, do_print=False, save_reservoir=False)

                joint = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test, do_print=False)[0]

                acc.append(joint)

            store_accuracies.append(np.mean(acc))

        syn_gain = store_accuracies[1] - store_accuracies[0]
        # syn_gain = np.trapz(store_accuracies - store_accuracies[0], dx=coupling_num)

        return syn_gain


    ##########################################


    # Load dataset

    X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

    # Simulations

    RUN_NUM = 10
    C_NUM = None

    # Bounds

    NNODE_MIN, NNODE_MAX = 100, 1000
    TRAIN_SIZE_MIN, TRAIN_SIZE_MAX = 0.5, 1.0

    # Initial solution
    nnode = 628 # random.randint(NNODE_MIN, NNODE_MAX)
    train_size = 0.586 # random.uniform(TRAIN_SIZE_MIN, TRAIN_SIZE_MAX)
    best_score = run_model(runs=RUN_NUM, coupling_num=C_NUM, r1_size=nnode, r2_size=nnode, train_sample=train_size)

    # Hill climbing loop
    MAX_ITER = 50
    NO_IMPROVEMENT_LIMIT = 51
    no_improvement_count = 0

    # Simulated annealing parameters
    T_init = 1.0       # Initial temperature
    T_min = 1e-3       # Minimum temperature
    alpha = np.exp(-0.95/MAX_ITER)       # Cooling rate

    T = T_init

    for i in range(MAX_ITER):
        # Start from current best
        new_nnode = nnode
        new_train_size = train_size

        # Temperature-scaled perturbation
        scale = T  # Larger changes at high T, smaller at low T

        # Propose a neighbor
        if random.choice([0, 1]) == 0:
            delta = int(nnode * random.uniform(-0.1, 0.1) * scale)
            new_nnode = max(NNODE_MIN, min(NNODE_MAX, nnode + delta))
        else:
            delta = train_size * random.uniform(-0.1, 0.1) * scale
            new_train_size = max(TRAIN_SIZE_MIN, min(TRAIN_SIZE_MAX, train_size + delta))

        # Evaluate
        new_score = run_model(runs=RUN_NUM, coupling_num=C_NUM, r1_size=new_nnode, r2_size=new_nnode, train_sample=new_train_size)

        # Accept solution or not
        delta_score = new_score - best_score

        if delta_score > 0:
            nnode, train_size = new_nnode, new_train_size
            best_score = new_score
            no_improvement_count = 0
            print(f"Iter {i}: Improved to score {best_score:.4f} with nnode={nnode}, train_size={train_size:.3f}")
        else:
            # Maybe accept worse solution
            scaled_delta = delta_score / abs(best_score) if best_score != 0 else delta_score
            prob = math.exp(scaled_delta / T)

            if random.random() < prob:
                nnode, train_size = new_nnode, new_train_size
                print(f"Iter {i}: Accepted worse score {new_score:.4f} (prob={prob:.4f})")
            else:
                no_improvement_count += 1
                print(f"Iter {i}: Rejected worse score {new_score:.4f} ({no_improvement_count} in a row)")

        # Cool down
        T = max(T_min, T * alpha)

        # Early stopping
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print(f"\nStopping early after {no_improvement_count} consecutive non-improving steps.")
            break

    print(f"\nBest found: score={best_score:.4f}, nnode={nnode}, train_size={train_size:.3f}")


    # Log end time
    end_time = time.time()
    elapsed = end_time - start_time
    formatted = str(timedelta(seconds=elapsed))
    print(f"Execution time: {formatted}")
