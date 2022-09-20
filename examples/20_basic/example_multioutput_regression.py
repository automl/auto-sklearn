# -*- encoding: utf-8 -*-
"""
=======================
Multi-output Regression
=======================

The following example shows how to fit a multioutput regression model with
*auto-sklearn*.
"""
import numpy as numpy
from pprint import pprint

from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from autosklearn.regression import AutoSklearnRegressor


############################################################################
# Data Loading
# ============

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

############################################################################
# Build and fit a regressor
# =========================

automl = AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_multioutput_regression_example_tmp",
)
automl.fit(X_train, y_train, dataset_name="synthetic")

############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())


############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("R2 score:", r2_score(y_test, predictions))

###########################################################################
# Get the configuration space
# ===========================

# The configuration space is reduced, i.e. no SVM.
print(automl.get_configuration_space(X_train, y_train))
