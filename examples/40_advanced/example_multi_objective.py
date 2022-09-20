# -*- encoding: utf-8 -*-
"""
==============
Classification
==============

The following example shows how to fit *auto-sklearn* to optimize for two
competing metrics: `precision` and `recall` (read more on this tradeoff
in the `scikit-learn docs <https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_.

Auto-sklearn uses `SMAC3's implementation of ParEGO <https://automl.github.io/SMAC3/main/details/multi_objective.html>`_.
Multi-objective ensembling and proper access to the full Pareto set will be added in the near
future.
"""
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import autosklearn.metrics


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=31, return_X_y=True, as_frame=True)
# Change the target to align with scikit-learn's convention that
# ``1`` is the minority class. In this example it is predicting
# that a credit is "bad", i.e. that it will default.
y = np.array([1 if val == "bad" else 0 for val in y])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    metric=[autosklearn.metrics.precision, autosklearn.metrics.recall],
    delete_tmp_folder_after_terminate=False,
)
automl.fit(X_train, y_train, dataset_name="German Credit")

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

############################################################################
# Visualize the Pareto set
# ==========================
plot_values = []
pareto_front = automl.get_pareto_set()
for ensemble in pareto_front:
    predictions = ensemble.predict(X_test)
    precision = sklearn.metrics.precision_score(y_test, predictions)
    recall = sklearn.metrics.recall_score(y_test, predictions)
    plot_values.append((precision, recall))
fig = plt.figure()
ax = fig.add_subplot(111)
for precision, recall in plot_values:
    ax.scatter(precision, recall, c="blue")
ax.set_xlabel("Precision")
ax.set_ylabel("Recall")
ax.set_title("Pareto set")
plt.show()
