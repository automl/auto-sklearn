# -*- encoding: utf-8 -*-
"""
=============
Feature Types
=============

In *auto-sklearn* it is possible to specify the feature types of a dataset when
calling the method :meth:`fit() <autosklearn.classification.AutoSklearnClassifier.fit>` by specifying the argument ``feat_type``.
The following example demonstrates a way it can be done.
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

try:
    import openml
except ImportError:
    print("#"*80 + """
    To run this example you need to install openml-python:

    pip install git+https://github.com/renatopp/liac-arff
    pip install requests xmltodict
    pip install git+https://github.com/openml/openml-python@develop --no-deps\n""" +
          "#"*80)
    raise


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
    feat_type = ['Categorical' if ci else 'Numerical'
                 for ci in categorical_indicator]

    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
    )
    cls.fit(X_train, y_train, feat_type=feat_type)

    predictions = cls.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
