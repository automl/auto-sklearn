import numpy as np
import sys

import sklearn.cross_validation


def split_data(X, Y):
    num_data_points = X.shape[0]
    X_train, X_valid, Y_train, Y_valid = None, None, None, None
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The first dimension of the X and Y array must "
                         "be equal.")
    try:
        sss = sklearn.cross_validation.StratifiedShuffleSplit(Y, n_iter=1,
                                                              test_size=0.33,
                                                              train_size=None,
                                                              random_state=42)
    except ValueError:
        sys.stdout.write("To few samples of one class or maybe a regression "
                         "dataset, use shuffle split.\n")
        sss = sklearn.cross_validation.ShuffleSplit(Y.shape[0], n_iter=1,
                                                    test_size=0.33,
                                                    train_size=None,
                                                    random_state=42)

    assert len(sss) == 1, "Splitting data went wrong"

    for train_index, valid_index in sss:
        X_train, X_valid = X[train_index], X[valid_index]
        Y_train, Y_valid = Y[train_index], Y[valid_index]

    assert X_train.shape[0] + X_valid.shape[0] == num_data_points
    assert Y_train.shape[0] + Y_valid.shape[0] == num_data_points

    return X_train, X_valid, Y_train, Y_valid


def get_CV_fold(X, Y, fold, folds, shuffle=True):
    num_data_points = X.shape[0]
    fold = int(fold)
    folds = int(folds)
    if fold >= folds:
        raise ValueError((fold, folds))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The first dimension of the X and Y array must "
                         "be equal.")

    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=folds,
                                                  shuffle=shuffle,
                                                  random_state=42)
    for idx, split in enumerate(kf):
        if idx == fold:
            break

    assert len(split[0]) + len(split[1]) == num_data_points

    return split
