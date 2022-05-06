# -*- encoding: utf-8 -*-
"""
==============
Classification
==============

The following example shows how to fit *auto-sklearn* to optimize for two
competing metrics: `precision` and `recall` (read more on this tradeoff
in the `scikit-learn docs <https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.

Auto-sklearn uses `SMAC3's implementation of ParEGO <https://automl.github.io/SMAC3/main/details/multi_objective.html>`_.
Multi-objective ensembling and proper access to the full Pareto front will be added in the near
future.
"""
from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    tmp_folder="/tmp/autosklearn_classification_example_tmp",
    metric=[autosklearn.metrics.precision, autosklearn.metrics.recall],
)
automl.fit(X_train, y_train, dataset_name="breast_cancer")

############################################################################
# Compute the two competing metrics
# =================================

predictions = automl.predict(X_test)
print("Precision", sklearn.metrics.precision_score(y_test, predictions))
print("Recall", sklearn.metrics.recall_score(y_test, predictions))

############################################################################
# View the models found by auto-sklearn
# =====================================
# They are by default sorted by the first metric given to *auto-sklearn*.

print(automl.leaderboard())

############################################################################
# ``cv_results`` also contains both metrics
# =========================================
# Similarly to the leaderboard, they are sorted by the first metric given
# to *auto-sklearn*.

pprint(automl.cv_results_)
