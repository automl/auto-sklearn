# -*- encoding: utf-8 -*-
"""
==========
Regression
==========

The following example shows how to fit a simple regression model with
*auto-sklearn*.
"""
from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.regression
import matplotlib.pyplot as plt

############################
# Data Loading
# ============

X, y = sklearn.datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

###########################
# Build and fit a regressor
# =========================

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_regression_example_tmp",
)
automl.fit(X_train, y_train, dataset_name="diabetes")

############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())

######################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

#####################################
# Get the Score of the final ensemble
# ===================================
# After training the estimator, we can now quantify the goodness of fit. One possibility for
# is the `R2 score <https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score>`_.
# The values range between -inf and 1 with 1 being the best possible value. A dummy estimator
# predicting the data mean has an R2 score of 0.

train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

######################
# Plot the predictions
# ====================
# Furthermore, we can now visually inspect the predictions. We plot the true value against the
# predictions and show results on train and test data. Points on the diagonal depict perfect
# predictions. Points below the diagonal were overestimated by the model (predicted value is higher
# than the true value), points above the diagonal were underestimated (predicted value is lower than
# the true value).

plt.scatter(train_predictions, y_train, label="Train samples", c="#d95f02")
plt.scatter(test_predictions, y_test, label="Test samples", c="#7570b3")
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.legend()
plt.plot([30, 400], [30, 400], c="k", zorder=0)
plt.xlim([30, 400])
plt.ylim([30, 400])
plt.tight_layout()
plt.show()
