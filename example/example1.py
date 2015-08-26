# -*- encoding: utf-8 -*-
from __future__ import print_function

import numpy as np

import autosklearn
import sklearn.datasets


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
    automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=10,
                                               per_run_time_limit=1,
                                               tmp_folder='/tmp/auto_tmp',
                                               output_folder='/tmp/auto_out',
                                               debug_mode=True,
                                               )
    automl.fit(X_train, y_train, dataset_name='example')
    # print(automl.score(X_test, y_test))


if __name__ == '__main__':
    main()
