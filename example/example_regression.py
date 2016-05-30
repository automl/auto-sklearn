# -*- encoding: utf-8 -*-
from __future__ import print_function

import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.cross_validation import train_test_split
import autosklearn.regression


def main():
    boston = sklearn.datasets.load_boston()
    X = boston.data
    y = boston.target
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=60, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_example_tmp',
        output_folder='/tmp/autosklearn_example_out')
    automl.fit(X_train, y_train, dataset_name='boston')

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))


if __name__ == '__main__':
    main()
