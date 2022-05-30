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


############################################################################
# Custom Metrics
# ==============
def accuracy(solution, prediction):
    # custom function defining accuracy
    return np.mean(solution == prediction)


def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def accuracy_wk(solution, prediction, extra_argument):
    # custom function defining accuracy and accepting an additional argument
    assert extra_argument is None
    return np.mean(solution == prediction)


def error_wk(solution, prediction, extra_argument):
    # custom function defining error and accepting an additional argument
    assert extra_argument is None
    return np.mean(solution != prediction)


def metric_which_needs_x(solution, prediction, X_data, consider_col, val_threshold):
    # custom function defining accuracy
    assert X_data is not None
    rel_idx = X_data[:, consider_col] > val_threshold
    return np.mean(solution[rel_idx] == prediction[rel_idx])


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

############################################################################
# Print a list of available metrics
# =================================

print("Available CLASSIFICATION metrics autosklearn.metrics.*:")
print("\t*" + "\n\t*".join(autosklearn.metrics.CLASSIFICATION_METRICS))

print("Available REGRESSION autosklearn.metrics.*:")
print("\t*" + "\n\t*".join(autosklearn.metrics.REGRESSION_METRICS))

############################################################################
# First example: Use predefined accuracy metric
# =============================================

print("#" * 80)
print("Use predefined accuracy metric")
scorer = autosklearn.metrics.accuracy
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    seed=1,
    metric=scorer,
)
cls.fit(X_train, y_train)

predictions = cls.predict(X_test)
score = scorer(y_test, predictions)
print(f"Accuracy score {score:.3f} using {scorer.name}")

############################################################################
# Second example: Use own accuracy metric
# =======================================

print("#" * 80)
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
    seed=1,
    metric=accuracy_scorer,
)
cls.fit(X_train, y_train)

predictions = cls.predict(X_test)
score = accuracy_scorer(y_test, predictions)
print(f"Accuracy score {score:.3f} using {accuracy_scorer.name:s}")

############################################################################
# Third example: Use own error metric
# ===================================

print("#" * 80)
print("Use self defined error metric")
error_rate = autosklearn.metrics.make_scorer(
    name="error",
    score_func=error,
    optimum=0,
    greater_is_better=False,
    needs_proba=False,
    needs_threshold=False,
)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    seed=1,
    metric=error_rate,
)
cls.fit(X_train, y_train)

cls.predictions = cls.predict(X_test)
score = error_rate(y_test, predictions)
print(f"Error score {score:.3f} using {error_rate.name:s}")

############################################################################
# Fourth example: Use own accuracy metric with additional argument
# ================================================================

print("#" * 80)
print("Use self defined accuracy with additional argument")
accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu_add",
    score_func=accuracy_wk,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    extra_argument=None,
)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60, per_run_time_limit=30, seed=1, metric=accuracy_scorer
)
cls.fit(X_train, y_train)

predictions = cls.predict(X_test)
score = accuracy_scorer(y_test, predictions)
print(f"Accuracy score {score:.3f} using {accuracy_scorer.name:s}")

############################################################################
# Fifth example: Use own accuracy metric with additional argument
# ===============================================================

print("#" * 80)
print("Use self defined error with additional argument")
error_rate = autosklearn.metrics.make_scorer(
    name="error_add",
    score_func=error_wk,
    optimum=0,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    extra_argument=None,
)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    seed=1,
    metric=error_rate,
)
cls.fit(X_train, y_train)

predictions = cls.predict(X_test)
score = error_rate(y_test, predictions)
print(f"Error score {score:.3f} using {error_rate.name:s}")


#############################################################################
# Sixth example: Use a metric with additional argument which also needs xdata
# ===========================================================================
"""
Finally, *Auto-sklearn* also support metric that require the train data (aka X_data) to
compute a value. This can be useful if one only cares about the score on a subset of the
data.
"""

accuracy_scorer = autosklearn.metrics.make_scorer(
    name="accu_X",
    score_func=metric_which_needs_x,
    optimum=1,
    greater_is_better=True,
    needs_proba=False,
    needs_X=True,
    needs_threshold=False,
    consider_col=1,
    val_threshold=18.8,
)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    seed=1,
    metric=accuracy_scorer,
)
cls.fit(X_train, y_train)

predictions = cls.predict(X_test)
score = metric_which_needs_x(
    y_test,
    predictions,
    X_data=X_test,
    consider_col=1,
    val_threshold=18.8,
)
print(f"Error score {score:.3f} using {accuracy_scorer.name:s}")
