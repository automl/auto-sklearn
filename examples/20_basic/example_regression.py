# -*- encoding: utf-8 -*-
"""
==========
Regression
==========

The following example shows how to fit a simple regression model with
*auto-sklearn*.
"""
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression
import matplotlib.pyplot as plt

############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Build and fit a regressor
# =========================

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
    output_folder='/tmp/autosklearn_regression_example_out',
)
automl.fit(X_train, y_train, dataset_name='diabetes')

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

print(automl.show_models())

###########################################################################
# Get the Score of the final ensemble
# ===================================

train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

###########################################################################
# Plot the predictions
# ===================================
plt.scatter(train_predictions, y_train, label="Train samples")
plt.scatter(test_predictions, y_test, label="Test samples")
plt.xlabel("Predictions")
plt.ylabel("True values")
plt.legend()
plt.plot([40, 350], [40, 350], c='k', zorder=0)
plt.xlim([40, 350])
plt.ylim([40, 350])
plt.show()