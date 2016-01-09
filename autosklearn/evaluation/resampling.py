# -*- encoding: utf-8 -*-
import numpy as np
import sklearn.cross_validation

from autosklearn.util import get_logger

logger = get_logger(__name__)


__all__ = [
    'split_data',
    'get_CV_fold'
]


def split_data(X, Y, classification=None):
    num_data_points = X.shape[0]
    num_labels = Y.shape[1] if len(Y.shape) > 1 else 1
    X_train, X_valid, Y_train, Y_valid = None, None, None, None
    if X.shape[0] != Y.shape[0]:
        raise ValueError('The first dimension of the X and Y array must '
                         'be equal.')

    # If one class only has one sample, put it into the training set
    if classification is True and num_labels == 1:
        classes, y_indices = np.unique(Y, return_inverse=True)
        if np.min(np.bincount(y_indices)) < 2:
            classes_with_one_sample = np.bincount(y_indices) < 2
            sample_idxs = []
            Y_old = Y
            indices = np.ones(Y.shape, dtype=bool)
            for i, class_with_one_sample in enumerate(classes_with_one_sample):
                if not class_with_one_sample:
                    continue
                sample_idx = np.argwhere(Y == classes[i])[0][0]
                indices[sample_idx] = False
                sample_idxs.append(sample_idx)
            Y = Y[indices]

    if num_labels > 1:
        logger.info('Multilabel dataset, do a random split.\n')
        sss = None
    else:
        try:
            sss = sklearn.cross_validation.StratifiedShuffleSplit(
                Y,
                n_iter=1,
                test_size=0.33,
                train_size=None,
                random_state=42)
        except ValueError:
            sss = None
            logger.info('Too few samples of one class or maybe a '
                        'regression dataset, use shuffle split.\n')

    if sss is None:
        sss = sklearn.cross_validation.ShuffleSplit(Y.shape[0],
                                                    n_iter=1,
                                                    test_size=0.33,
                                                    train_size=None,
                                                    random_state=42)

    assert len(sss) == 1, 'Splitting data went wrong'

    for train_index, valid_index in sss:
        if classification is True and num_labels == 1:
            try:
                Y = Y_old
                for sample_idx in sorted(sample_idxs):
                    train_index[train_index >= sample_idx] += 1
                    valid_index[valid_index >= sample_idx] += 1
                for sample_idx in sample_idxs:
                    train_index = np.append(train_index, np.array(sample_idx))
            except UnboundLocalError:
                pass

        X_train, X_valid = X[train_index], X[valid_index]
        Y_train, Y_valid = Y[train_index], Y[valid_index]

    assert X_train.shape[0] + X_valid.shape[0] == num_data_points
    assert Y_train.shape[0] + Y_valid.shape[0] == num_data_points

    return X_train, X_valid, Y_train, Y_valid


def get_CV_fold(X, Y, fold, folds, shuffle=True, random_state=None):
    num_data_points = X.shape[0]
    fold = int(fold)
    folds = int(folds)
    if fold >= folds:
        raise ValueError((fold, folds))
    if X.shape[0] != Y.shape[0]:
        raise ValueError('The first dimension of the X and Y array must '
                         'be equal.')

    if len(Y.shape) > 1:
        kf = sklearn.cross_validation.KFold(n=Y.shape[0], n_folds=folds,
                                            shuffle=shuffle,
                                            random_state=random_state)
    else:
        kf = sklearn.cross_validation.StratifiedKFold(Y,
                                                      n_folds=folds,
                                                      shuffle=shuffle,
                                                      random_state=random_state)
    for idx, split in enumerate(kf):
        if idx == fold:
            break

    assert len(split[0]) + len(split[1]) == num_data_points

    return split
