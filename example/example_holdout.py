# -*- encoding: utf-8 -*-
from __future__ import print_function
from operator import itemgetter

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


# Utility function to report best scores
# from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


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
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_holdout_example_tmp',
        output_folder='/tmp/autosklearn_holdout_example_out')
    automl.fit(X_train, y_train, dataset_name='digits')

    # Print the best models together with their scores - if all scores are
    # unreasonably bad (around 0.0) you should have a look into the logging
    # file to figure out the error
    report(automl.grid_scores_)
    print(automl.show_models())
    predictions = automl.predict(X_test)
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
