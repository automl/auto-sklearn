# -*- encoding: utf-8 -*-


"""
================
Cross-Validation
================

In *auto-sklearn* it is possible to use different resampling strategies
by specifying the arguments ``resampling_strategy`` and
``resampling_strategy_arguments``. The following example shows how to use
cross-validation and how to set the folds when instantiating
``AutoSklearnClassifier``.
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Building  and fitting the classifier
# ====================================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_cv_example_tmp',
    output_folder='/tmp/autosklearn_cv_example_out',
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
)

# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy(), dataset_name='breast_cancer')

############################################################################
# Print Results before refit
# ==========================
print(automl.sprint_statistics())

# One can use models trained during cross-validation directly to predict
# for unseen data. For this, all k models trained during k-fold
# cross-validation are considered as a single soft-voting ensemble inside
# the ensemble constructed with ensemble selection.
print('Before re-fit')
predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

############################################################################
# Perform a refit
# ===============
# During fit(), models are fit on individual cross-validation folds. To use
# all available data, we call refit() which trains all models in the
# final ensemble on the whole dataset.
automl.refit(X_train.copy(), y_train.copy())
predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
