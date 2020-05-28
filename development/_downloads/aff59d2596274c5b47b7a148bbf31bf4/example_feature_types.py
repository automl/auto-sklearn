# -*- encoding: utf-8 -*-
"""
=============
Feature Types
=============

In *auto-sklearn* it is possible to specify the feature types of a dataset when calling the method
:meth:`fit() <autosklearn.classification.AutoSklearnClassifier.fit>` by specifying the argument
``feat_type``. The following example demonstrates a way it can be done.
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn import preprocessing

import autosklearn.classification


############################################################################
# Data Loading
# ============
# Load adult dataset from openml.org, see https://www.openml.org/t/2117
X, y = sklearn.datasets.fetch_openml(data_id=179, return_X_y=True)

# y needs to be encoded, as fetch openml doesn't download a float
y = preprocessing.LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = \
     sklearn.model_selection.train_test_split(X, y, random_state=1)

# Create feature type list from openml.org indicator and run autosklearn
data = sklearn.datasets.fetch_openml(data_id=179, as_frame=True)
feat_type = ['Categorical' if x.name == 'category' else 'Numerical' for x in data['data'].dtypes]

############################################################################
# Build and fit a classifier
# ==========================

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
)
cls.fit(X_train, y_train, feat_type=feat_type)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = cls.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
