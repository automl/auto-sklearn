# -*- encoding: utf-8 -*-
"""
==========================
Fit a single configuration
==========================

*Auto-sklearn* searches for the best combination of machine learning algorithms
and their hyper-parameter configuration for a given task, using Scikit-Learn Pipelines.
To further improve performance, this pipelines are ensemble together using Ensemble
Selection from Caruana (2004).


This example shows how one can fit one of this pipelines, both, with an user defined
configuration, and a randomly sampled one form the configuration space.

The pipelines that Auto-Sklearn fits are compatible with Scikit-Learn API. You can
get further documentation about Scikit-Learn models here: <https://scikit-learn.org/stable/getting_started.html`>_
"""
import numpy as np
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from ConfigSpace.configuration_space import Configuration

import autosklearn.classification


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=3, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.5, random_state=3
)

############################################################################
# Define an estimator
# ============================

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=60,
    memory_limit=4096,
    # We will limit the configuration space only to
    # have RandomForest as a valid model. We recommend enabling all
    # possible models to get a better performance.
    include_estimators=['random_forest'],
    delete_tmp_folder_after_terminate=False,
)

###########################################################################
# Fit an user provided configuration
# ==================================

# We will create a configuration that has a user defined
# min_samples_split in the Random Forest. We recommend you to look into
# how the ConfigSpace package works here:
# https://automl.github.io/ConfigSpace/master/
cs = cls.get_configuration_space(X, y, dataset_name='kr-vs-kp')
config = cs.sample_configuration()
config._values['classifier:random_forest:min_samples_split'] = 11

# Make sure that your changed configuration complies with the configuration space
config.is_valid_configuration()

pipeline, run_info, run_value = cls.fit_pipeline(X=X_train, y=y_train,
                                                 dataset_name='kr-vs-kp',
                                                 config=config,
                                                 X_test=X_test, y_test=y_test)

# This object complies with Scikit-Learn Pipeline API.
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
print(pipeline.named_steps)

# The fit_pipeline command also returns a named tuple with the pipeline constraints
print(run_info)

# The fit_pipeline command also returns a named tuple with train/test performance
print(run_value)

# We can make sure that our pipeline configuration was honored as follows
print("Passed Configuration:", pipeline.config)
print("Random Forest:", pipeline.named_steps['classifier'].choice.estimator)

# We can also search for new configurations using the fit() method
# Any configurations found by Auto-Sklearn -- even the ones created using
# fit_pipeline() are stored to disk and can be used for Ensemble Selection
cs = cls.fit(X, y, dataset_name='kr-vs-kp')
