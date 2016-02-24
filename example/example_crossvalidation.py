# -*- encoding: utf-8 -*-
from __future__ import print_function

import sklearn.datasets
import numpy as np

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
        output_folder='/tmp/autosklearn_example_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv', resampling_strategy_arguments={'folds': 5})

    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    automl.fit(X_train.copy(), y_train.copy(), dataset_name='digits')
    automl.refit(X_train.copy(), y_train.copy())

    print(automl.show_models())

    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
