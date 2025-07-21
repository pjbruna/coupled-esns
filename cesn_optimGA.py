import numpy as np
import random
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import japanese_vowels
from cesn_model import *


rpy.verbosity(0)

seed = 42
random.seed(seed)


# Redirect stdout and stderr to a file
with open("INVERT_genetic_log.txt", "w", buffering=1) as f:
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
            acc = []

            for r in range(runs):
                model = CesnModel(r1_nnodes=r1_size, r2_nnodes=r2_size, coupling_strength=c)

                model.train_r1(input=X_train1, target=Y_train1)
                model.train_r2(input=X_train2, target=Y_train2)

                results = model.test(input=X_test, target=Y_test, noise_scale=0, do_print=False, save_reservoir=False)

                joint = model.accuracy(pred1=results[0], pred2=results[1], target=Y_test, do_print=False)[0]

                acc.append(joint)

            store_accuracies.append(np.mean(acc))

            # print(f'N: {runs}; CS: {c}; R1: {r1_size}; R2: {r2_size};  Acc: {np.mean(acc)}')

        gain = store_accuracies[0] - store_accuracies[1] # maximize woc over coupling
        # gain = store_accuracies[1] - store_accuracies[0] # maximize coupling over woc

        # gain = np.trapz(store_accuracies - store_accuracies[0], dx=coupling_num)

        return gain



    ##########################################



    # Individual representation: (nnode, train_size)
    def create_individual():
        return [random.randint(NNODE_MIN, NNODE_MAX), random.uniform(TRAIN_SIZE_MIN, TRAIN_SIZE_MAX)]

    # Fitness function
    def fitness(individual):
        print(individual)
        nnode, train_size = individual
        return run_model(runs=1, coupling_num=None, r1_size=nnode, r2_size=nnode, train_sample=train_size)

    # Mutation
    def mutate(individual):
        if random.random() < MUTATION_RATE:
            individual[0] = int(individual[0] * (1 + random.choice(pdelta)))
            individual[0] = max(NNODE_MIN, min(NNODE_MAX, individual[0]))
        if random.random() < MUTATION_RATE:
            individual[1] = individual[1] * (1 + random.choice(pdelta))
            individual[1] = max(TRAIN_SIZE_MIN, min(TRAIN_SIZE_MAX, individual[1]))
        return individual

    # Crossover (uniform)
    def crossover(parent1, parent2):
        child = [
            parent1[0] if random.random() < 0.5 else parent2[0],
            parent1[1] if random.random() < 0.5 else parent2[1]
        ]
        return child

    # Selection (tournament)
    def select(population, scores, k=2):
        selected = random.sample(list(zip(population, scores)), k)
        return max(selected, key=lambda x: x[1])[0]



    ####################################



    # Load dataset

    X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

    # Define bounds

    NNODE_MIN, NNODE_MAX = 50, 1000 # 100, 1000
    TRAIN_SIZE_MIN, TRAIN_SIZE_MAX = 0.5, 1.0

    # Parameters

    POP_SIZE = 30 # 20
    GENERATIONS = 100 # 50
    MUTATION_RATE = 0.3 # 0.2
    pdelta = [-0.1, 0, 0.1] # [-0.1, -0.09, -0.08, 0, 0.08, 0.09, 0.1] 

    # Main GA loop
    population = [create_individual() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        scores = [fitness(ind) for ind in population]
        next_gen = []

        for _ in range(POP_SIZE):
            parent1 = select(population, scores)
            parent2 = select(population, scores)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen
        best_score = max(scores)
        best_individual = population[scores.index(best_score)]
        print(f"Gen {gen}: Best Score = {best_score:.4f}, Params = {best_individual}")


    # Log end time
    end_time = time.time()
    elapsed = end_time - start_time
    formatted = str(timedelta(seconds=elapsed))
    print(f"Execution time: {formatted}")