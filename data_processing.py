import os
import numpy as np
import pandas as pd
from reservoirpy.datasets import japanese_vowels


def print_dataset_info(
    train_class_count, test_class_count, 
    exemplars_train, exemplars_test, 
    X_train, X_train_balanced, 
    X_test, X_test_balanced, 
    n_classes
):
    
    train_has_all = set(np.unique(train_class_count)) == set(np.arange(n_classes))
    test_has_all  = set(np.unique(test_class_count)) == set(np.arange(n_classes))

    rows = [
        ("Includes all classes", train_has_all, test_has_all),
        ("Exemplars per class", exemplars_train, exemplars_test),
        ("# Dropped signals", len(X_train) - len(X_train_balanced), len(X_test) - len(X_test_balanced)),
    ]

    # header
    print("="*45)
    print(f"{'Summary':25} | {'Train':^7} | {'Test':^7}")
    print("-"*45)

    # rows
    for metric, train_val, test_val in rows:
        print(f"{metric:25} | {str(train_val):^7} | {str(test_val):^7}")

    print("="*45)


def generate_jvowels(signal_length=None, do_print=False):
    # load data
    X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

    # count classes
    n_classes = Y_train[0].shape[1]

    # replace with [1,-1/8] coding-scheme
    Y_train = [np.where(signal == 0, -1/8, signal) for signal in Y_train]
    Y_test = [np.where(signal == 0, -1/8, signal) for signal in Y_test]

    # discard signals shorter than signal_length
    trunc = signal_length

    train_keep_idx = [i for i, signal in enumerate(X_train) if signal.shape[0] >= trunc]
    test_keep_idx  = [i for i, signal in enumerate(X_test)  if signal.shape[0] >= trunc]

    X_train_filtered = [X_train[i] for i in train_keep_idx]
    Y_train_filtered = [Y_train[i] for i in train_keep_idx]

    X_test_filtered  = [X_test[i]  for i in test_keep_idx]
    Y_test_filtered  = [Y_test[i]  for i in test_keep_idx]

    # truncate remaining signals
    X_train_filtered = [signal[:trunc, :] for signal in X_train_filtered]
    Y_train_filtered = [signal[:trunc, :] for signal in Y_train_filtered]

    X_test_filtered  = [signal[:trunc, :] for signal in X_test_filtered]
    Y_test_filtered  = [signal[:trunc, :] for signal in Y_test_filtered]

    # tally remaining signals per class
    train_class_count = [np.argmax(signal[0,:]) for signal in Y_train_filtered]
    test_class_count = [np.argmax(signal[0,:]) for signal in Y_test_filtered]

    # sample uniformly per class
    exemplars_train = np.min(np.unique(train_class_count, return_counts=True)[1])
    exemplars_test = np.min(np.unique(test_class_count, return_counts=True)[1])

    X_train_balanced, Y_train_balanced = [], []
    for cls in range(n_classes):
        idx = [i for i, signal in enumerate(Y_train_filtered) if np.argmax(signal[0]) == cls]
        chosen = np.random.choice(idx, size=int(exemplars_train), replace=False)
        X_train_balanced.extend([X_train_filtered[i] for i in chosen])
        Y_train_balanced.extend([Y_train_filtered[i] for i in chosen])

    X_test_balanced, Y_test_balanced = [], []
    for cls in range(n_classes):
        idx = [i for i, signal in enumerate(Y_test_filtered) if np.argmax(signal[0]) == cls]
        chosen = np.random.choice(idx, size=int(exemplars_test), replace=False)
        X_test_balanced.extend([X_test_filtered[i] for i in chosen])
        Y_test_balanced.extend([Y_test_filtered[i] for i in chosen])

    if do_print==True:
        print_dataset_info(
            train_class_count, test_class_count, 
            exemplars_train, exemplars_test, 
            X_train, X_train_balanced, 
            X_test, X_test_balanced, 
            n_classes
        )

    return X_train_balanced, Y_train_balanced, X_test_balanced, Y_test_balanced


def save_progress(results_list, path, final=False):
    df = pd.DataFrame(results_list)
    if final:
        df.to_csv(path, index=False)
        print(f"Final results saved to {path} ({len(df)} entries).")
    else:
        df.to_csv(
            path,
            mode='a',
            header=not os.path.exists(path),
            index=False
        )
        print(f"Progress saved ({len(df)} total results).")
