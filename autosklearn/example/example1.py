# -*- encoding: utf-8 -*-
from __future__ import print_function
import pprint

import numpy as np

import autosklearn
import sklearn.datasets


def debug_fit(automl, X_train, y_train):
    automl.fit(X_train, y_train, dataset_name='example')

def norm_fit(automl,  X_train, y_train):
    automl.fit(X_train, y_train)

def main():
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train = X[:1000]
    y_train = y[:1000]
    X_test = X[1000:]
    y_test = y[1000:]
    debug_mode = True

    pprint.pprint(len(X_train))
    pprint.pprint(y_train)
    #
    # if debug_mode:
    #     automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=10,
    #                                        per_run_time_limit=1,
    #                                        tmp_folder='/tmp/auto_tmp',
    #                                        output_folder='/tmp/auto_out',
    #                                        debug_mode=True)
    # else:
    #     automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=10,
    #                                per_run_time_limit=1)
    #
    # if debug_mode:
    #     debug_fit(automl, X_train, y_train)
    # else:
    #     norm_fit(automl, X_train, y_train)
    #
    # print(automl.score(X_test, y_test))


if __name__ == '__main__':
    main()
