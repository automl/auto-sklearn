# -*- encoding: utf-8 -*-
from __future__ import print_function

import numpy as np
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


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
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_example_tmp',
        output_folder='/tmp/autosklearn_example_out')
    automl.fit(X_train, y_train, dataset_name='digits')

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
