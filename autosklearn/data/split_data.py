import numpy as np
import sys

import sklearn.cross_validation


def split_data(X, Y):
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
                         "dataset, use shuffle split")
        sss = sklearn.cross_validation.ShuffleSplit(Y.shape[0], n_iter=1,
                                                    test_size=0.33,
                                                    train_size=None,
                                                    random_state=42)

    assert len(sss) == 1, "Splitting data went wrong"

    for train_index, valid_index in sss:
        X_train, X_valid = X[train_index], X[valid_index]
        Y_train, Y_valid = Y[train_index], Y[valid_index]

    return X_train, X_valid, Y_train, Y_valid


def get_CV_fold(X, Y, fold, folds, shuffle=True):
    fold = int(fold)
    folds = int(folds)
    if fold >= folds:
        raise ValueError((fold, folds))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The first dimension of the X and Y array must "
                         "be equal.")

    if shuffle == True:
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        rs.shuffle(indices)
        Y = Y[indices]

    kf = sklearn.cross_validation.StratifiedKFold(Y, n_folds=folds)
    for idx, split in enumerate(kf):
        if idx == fold:
            break

    if shuffle == True:
        return indices[split[0]], indices[split[1]]
    return split
