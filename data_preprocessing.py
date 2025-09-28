import numpy as np
from reservoirpy.datasets import japanese_vowels


def generate_jvowels(signal_length=None, do_print=False):
    # load data
    X_train, Y_train, X_test, Y_test = japanese_vowels(repeat_targets=True)

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
    for cls in range(Y_train_filtered[0].shape[1]):
        idx = [i for i, signal in enumerate(Y_train_filtered) if np.argmax(signal[0]) == cls]
        chosen = np.random.choice(idx, size=int(exemplars_train), replace=False)
        X_train_balanced.extend([X_train_filtered[i] for i in chosen])
        Y_train_balanced.extend([Y_train_filtered[i] for i in chosen])

    X_test_balanced, Y_test_balanced = [], []
    for cls in range(Y_test_filtered[0].shape[1]):
        idx = [i for i, signal in enumerate(Y_test_filtered) if np.argmax(signal[0]) == cls]
        chosen = np.random.choice(idx, size=int(exemplars_test), replace=False)
        X_test_balanced.extend([X_test_filtered[i] for i in chosen])
        Y_test_balanced.extend([Y_test_filtered[i] for i in chosen])

    if do_print==True:
        print(f'Includes all classes (train): {np.all(np.unique(train_class_count, return_counts=True)[0] == np.array(range(Y_train_filtered[0].shape[1])))}')
        print(f'Includes all classes (test): {np.all(np.unique(test_class_count, return_counts=True)[0] == np.array(range(Y_train_filtered[0].shape[1])))}')
        print(f'Dropped signals (train): {len(X_train) - len(X_train_filtered)}')
        print(f'Dropped signals (test): {len(X_test) - len(X_train_filtered)}')
        print(f'Exemplars per class (train): {exemplars_train}')
        print(f'Exemplars per class (test): {exemplars_test} \n')

    return X_train_balanced, Y_train_balanced, X_test_balanced, Y_test_balanced