# -*- encoding: utf-8 -*-
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

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

    cls = autosklearn.classification.\
        AutoSklearnClassifier(time_left_for_this_task=120,
                              per_run_time_limit=30)
    cls.fit(X_train, y_train, feat_type=feat_type)

    predictions = cls.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
