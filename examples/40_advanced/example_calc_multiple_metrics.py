# -*- encoding: utf-8 -*-
"""
=======
Metrics
=======

In *Auto-sklearn*, model is optimized over a metric, either built-in or
custom metric. Moreover, it is also possible to calculate multiple metrics
per run. The following examples show how to calculate metrics built-in
and self-defined metrics for a classification problem.
"""

import autosklearn.classification
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from autosklearn.metrics import balanced_accuracy, precision, recall, f1


def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def get_metric_result(cv_results):
    results = pd.DataFrame.from_dict(cv_results)
    results = results[results['status'] == "Success"]
    cols = ['rank_test_scores', 'param_classifier:__choice__', 'mean_test_score']
    cols.extend([key for key in cv_results.keys() if key.startswith('metric_')])
    return results[cols]


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Build and fit a classifier
# ==========================

error_rate = autosklearn.metrics.make_scorer(
    name='custom_error',
    score_func=error,
    optimum=0,
    greater_is_better=False,
    needs_proba=False,
    needs_threshold=False
)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    scoring_functions=[balanced_accuracy, precision, recall, f1, error_rate]
)
cls.fit(X_train, y_train, X_test, y_test)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = cls.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

print("#" * 80)
print("Metric results")
print(get_metric_result(cls.cv_results_).to_string(index=False))
