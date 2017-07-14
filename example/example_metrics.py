# -*- encoding: utf-8 -*-
import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics



def accuracy(solution, prediction):
    # function defining accuracy
    return np.mean(solution == prediction)


def accuracy_wk(solution, prediction, dummy):
    # function defining accuracy and accepting an additional argument
    assert dummy is None
    return np.mean(solution == prediction)


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
    cls = autosklearn.classification.\
        AutoSklearnClassifier(time_left_for_this_task=60,
                              per_run_time_limit=30, seed=1)
    cls.fit(X_train, y_train, metric=autosklearn.metrics.accuracy)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl._automl._metric.name))

    # Second example: Use own accuracy metric
    print("#"*80)
    print("Use self defined accuracy metric")
    accuracy_scorer = autosklearn.metrics.make_scorer(name="accu",
                                                      score_func=accuracy,
                                                      greater_is_better=True,
                                                      needs_proba=False,
                                                      needs_threshold=False)
    cls = autosklearn.classification.\
        AutoSklearnClassifier(time_left_for_this_task=60,
                              per_run_time_limit=30, seed=1)
    cls.fit(X_train, y_train, metric=accuracy_scorer)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl._automl._metric.name))

    # Third example: Use own accuracy metric with additional argument
    print("#"*80)
    print("Use self defined accuracy with additional argument")
    accuracy_scorer = autosklearn.metrics.make_scorer(name="accu_add",
                                                      score_func=accuracy_wk,
                                                      greater_is_better=True,
                                                      needs_proba=False,
                                                      needs_threshold=False,
                                                      dummy=None)
    cls = autosklearn.classification.\
        AutoSklearnClassifier(time_left_for_this_task=60,
                              per_run_time_limit=30, seed=1)
    cls.fit(X_train, y_train, metric=accuracy_scorer)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl._automl._metric.name))


if __name__ == "__main__":
    main()
