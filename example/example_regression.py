# -*- encoding: utf-8 -*-
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression


def main():
    boston = sklearn.datasets.load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out')
    automl.fit(X_train, y_train, dataset_name='boston')

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))


if __name__ == '__main__':
    main()
