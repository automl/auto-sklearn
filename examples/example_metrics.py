# -*- encoding: utf-8 -*-
"""
=======
Metrics
=======

*Auto-sklearn* supports various built-in metrics, which can be found in the
:ref:`metrics section in the API <api:Built-in Metrics>`. However, it is also
possible to define your own metric and use it to fit and evaluate your model.
The following examples show how to use built-in and self-defined metrics for a
classification problem.
"""

import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics


def accuracy(solution, prediction):
    # custom function defining accuracy
    return np.mean(solution == prediction)


def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def accuracy_wk(solution, prediction, dummy):
    # custom function defining accuracy and accepting an additional argument
    assert dummy is None
    return np.mean(solution == prediction)


def error_wk(solution, prediction, dummy):
    # custom function defining error and accepting an additional argument
    assert dummy is None
    return np.mean(solution != prediction)


def main():

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Print a list of available metrics
    print("Available CLASSIFICATION metrics autosklearn.metrics.*:")
    print("\t*" + "\n\t*".join(autosklearn.metrics.CLASSIFICATION_METRICS))

    print("Available REGRESSION autosklearn.metrics.*:")
    print("\t*" + "\n\t*".join(autosklearn.metrics.REGRESSION_METRICS))

    # First example: Use predefined accuracy metric
    print("#"*80)
    print("Use predefined accuracy metric")
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        seed=1,
    )
    cls.fit(X_train, y_train, metric=autosklearn.metrics.accuracy)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl[0]._metric.name))

    # Second example: Use own accuracy metric
    print("#"*80)
    print("Use self defined accuracy metric")
    accuracy_scorer = autosklearn.metrics.make_scorer(
        name="accu",
        score_func=accuracy,
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
    )
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        seed=1,
    )
    cls.fit(X_train, y_train, metric=accuracy_scorer)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl[0]._metric.name))

    print("#"*80)
    print("Use self defined error metric")
    error_rate = autosklearn.metrics.make_scorer(
        name='error',
        score_func=error,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False
    )
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        seed=1
    )
    cls.fit(X_train, y_train, metric=error_rate)

    cls.predictions = cls.predict(X_test)
    print("Error rate {:g} using {:s}".
          format(error_rate(y_test, predictions),
                 cls._automl[0]._metric.name))

    # Third example: Use own accuracy metric with additional argument
    print("#"*80)
    print("Use self defined accuracy with additional argument")
    accuracy_scorer = autosklearn.metrics.make_scorer(
        name="accu_add",
        score_func=accuracy_wk,
        optimum=1,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
        dummy=None,
    )
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        seed=1,
    )
    cls.fit(X_train, y_train, metric=accuracy_scorer)

    predictions = cls.predict(X_test)
    print(
        "Accuracy score {:g} using {:s}".format(
            sklearn.metrics.accuracy_score(y_test, predictions),
            cls._automl[0]._metric.name
        )
    )

    print("#"*80)
    print("Use self defined error with additional argument")
    error_rate = autosklearn.metrics.make_scorer(
        name="error_add",
        score_func=error_wk,
        optimum=0,
        greater_is_better=True,
        needs_proba=False,
        needs_threshold=False,
        dummy=None,
    )
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        seed=1,
    )
    cls.fit(X_train, y_train, metric=error_rate)

    predictions = cls.predict(X_test)
    print(
        "Error rate {:g} using {:s}".format(
            error_rate(y_test, predictions),
            cls._automl[0]._metric.name
        )
    )


if __name__ == "__main__":
    main()
