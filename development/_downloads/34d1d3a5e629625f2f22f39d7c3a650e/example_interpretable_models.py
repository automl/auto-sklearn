# -*- encoding: utf-8 -*-
"""
====================
Interpretable models
====================

The following example shows how to inspect the models which *auto-sklearn*
optimizes over and how to restrict them to an interpretable subset.
"""
from pprint import pprint

import autosklearn.classification
import sklearn.datasets
import sklearn.metrics

############################################################################
# Show available classification models
# ====================================
#
# We will first list all classifiers Auto-sklearn chooses from. A similar
# call is available for preprocessors (see below) and regression (not shown)
# as well.
from autosklearn.pipeline.components.classification import ClassifierChoice

for name in ClassifierChoice.get_components():
    print(name)

############################################################################
# Show available preprocessors
# ============================

from autosklearn.pipeline.components.feature_preprocessing import (
    FeaturePreprocessorChoice,
)

for name in FeaturePreprocessorChoice.get_components():
    print(name)

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
#
# We will now only use a subset of the given classifiers and preprocessors.
# Furthermore, we will restrict the ensemble size to ``1`` to only use the
# single best model in the end. However, we would like to note that the
# choice of which models is deemed interpretable is very much up to the user
# and can change from use case to use case.

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_interpretable_models_example_tmp",
    include={
        "classifier": ["decision_tree", "lda", "sgd"],
        "feature_preprocessor": [
            "no_preprocessing",
            "polynomial",
            "select_percentile_classification",
        ],
    },
    ensemble_kwargs={"ensemble_size": 1},
)
automl.fit(X_train, y_train, dataset_name="breast_cancer")

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
