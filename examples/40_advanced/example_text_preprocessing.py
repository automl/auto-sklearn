# -*- encoding: utf-8 -*-
"""
==================
Text Preprocessing
==================
This example shows, how to use text features in *auto-sklearn*. *auto-sklearn*
can automatically encode text features if they are provided as string type.

For processing text features you need a pandas dataframe and set the desired
text columns to string and the categorical columns to category.

*auto-sklearn* ass text embedding creates a bag of words count
(https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from smac.tae import StatusType

import autosklearn.classification

############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=40945, return_X_y=True)

# on default the string columns is not assigned to the stirng features
print(f"{X.info()}\n")

# manuelly label all string columns
X = X.astype({'name': 'string', 'ticket': 'string', 'cabin': 'string', 'boat': 'string',
              'home.dest': 'string'})

# now *auto-sklearn* handles the string columns with it text feature preprocessing pipeline

X_train, X_test, y_train, y_test = \
     sklearn.model_selection.train_test_split(X, y, random_state=1)

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={'runcount_limit': 1},
)

cls.fit(X_train, y_train, X_test, y_test)

predictions = cls.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


X, y = sklearn.datasets.fetch_openml(data_id=40945, return_X_y=True, as_frame=True)
X = X.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = \
     sklearn.model_selection.train_test_split(X, y, random_state=1)

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={'runcount_limit': 1},
)

cls.fit(X_train, y_train, X_test, y_test)

predictions = cls.predict(X_test)
print("Accuracy score without text preprocessing", sklearn.metrics.accuracy_score(y_test, predictions))
