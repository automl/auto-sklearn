# -*- encoding: utf-8 -*-
"""
===============================
Test and Train data with Pandas
===============================

*auto-sklearn* can automatically encode categorical columns using a label/ordinal encoder. This example highlights how to properly set the dtype in a DataFrame for this to happen, and showcase how to input also testing data to autosklearn.
"""
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn import preprocessing

import autosklearn.classification


############################################################################
# Data Loading
# ============

# Using Australian dataset https://www.openml.org/d/40981
X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=True)

# bool and category will be automatically encoded.
# Targets for classification are also automatically encoded
# If using fetch_openml, data is already properly encoded, below
# is an example for user reference
desired_boolean_columns = ['A1']
desired_categorical_columns = ['A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
desired_numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
for column in X.columns:
    if column in desired_boolean_columns:
        X[column] = X[column].astype('bool')
    elif column in desired_categorical_columns:
        X[column] = X[column].astype('category')
    else:
        X[column] = pd.to_numeric(X[column])

y = pd.DataFrame(y, dtype='category')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.5, random_state=1
)

############################################################################
# Build and fit a classifier
# ==========================

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
)
cls.fit(X_train, y_train, X_test, y_test)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = cls.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

############################################################################
# Plot the ensemble performance
# ===================================

ensemble_performance_frame = pd.DataFrame(cls._automl[0].ensemble_performance_history)
ensemble_performance_frame.plot(
    x='Timestamp',
    y=['train_score', 'test_score'],
    kind='line',
    legend=True,
    title='Ensemble accuracy over time',
    grid=True,
)
plt.show()
