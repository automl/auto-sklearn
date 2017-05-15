# -*- encoding: utf-8 -*-
import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics

try:
    import openml
except ImportError:
    print("#"*80 + """
    To run this example you need to install openml-python:

    git+https://github.com/renatopp/liac-arff
    # OpenML is currently not on pypi, use an old version to not depend on
    # scikit-learn 0.18
    requests
    xmltodict
    git+https://github.com/renatopp/liac-arff
    git+https://github.com/openml/""" +
    "openml-python@0b9009b0436fda77d9f7c701bd116aff4158d5e1\n""" +
          "#"*80)
    raise


def accuracy(solution, prediction):
    # function defining accuracy
    return np.mean(solution == prediction)


def accuracy_wk(solution, prediction, dummy):
    # function defining accuracy and accepting an additional argument
    assert dummy is None
    return np.mean(solution == prediction)


def main():
    # Load adult dataset from openml.org, see https://www.openml.org/t/2117
    openml.config.apikey = '610344db6388d9ba34f6db45a3cf71de'

    task = openml.tasks.get_task(2117)
    train_indices, test_indices = task.get_train_test_split_indices()
    X, y = task.get_X_and_y()

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    dataset = task.get_dataset()
    _, _, categorical_indicator = dataset.\
        get_data(target=task.target_name, return_categorical_indicator=True)

    # Create feature type list from openml.org indicator and run autosklearn
    feat_type = ['categorical' if ci else 'numerical'
                 for ci in categorical_indicator]

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
    cls.fit(X_train, y_train, feat_type=feat_type,
            metric=autosklearn.metrics.accuracy)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl._automl._metric.name))

    print("#"*80)
    print("Use self defined accuracy accuracy metric")
    accuracy_scorer = autosklearn.metrics.make_scorer(name="accu",
                                                      score_func=accuracy,
                                                      greater_is_better=True,
                                                      needs_proba=False,
                                                      needs_threshold=False)
    cls = autosklearn.classification.\
        AutoSklearnClassifier(time_left_for_this_task=60,
                              per_run_time_limit=30, seed=1)
    cls.fit(X_train, y_train, feat_type=feat_type, metric=accuracy_scorer)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl._automl._metric.name))

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
    cls.fit(X_train, y_train, feat_type=feat_type, metric=accuracy_scorer)

    predictions = cls.predict(X_test)
    print("Accuracy score {:g} using {:s}".
          format(sklearn.metrics.accuracy_score(y_test, predictions),
                 cls._automl._automl._metric.name))


if __name__ == "__main__":
    main()
