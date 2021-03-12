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

X, y = sklearn.datasets.fetch_openml(data_id=3, return_X_y=True, as_frame=False)
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
)

###########################################################################
# Fit a random pipeline
# =====================

# First we fit a pipeline, product of a random configuration
# The constrains set on the AutoSklearnClassifier will be honored.
# In this case we requested to use 4096 Mb as a memory limit
pipeline, run_info, run_value = cls.fit_pipeline(X=X_train, y=y_train,
                                                 X_test=X_test, y_test=y_test)

# This object complies with Scikit-Learn Pipeline API.
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
print(pipeline.named_steps)

# The fit_pipeline command also returns a named tuple with the pipeline constraints
# Notice the random configuration that was sampled from the configuration space
print(run_info)

# The fit_pipeline command also returns a named tuple with the run results, for
# example, run duration, train loss and test loss. Auto-Sklearn solves a minimization
# problem, so metrics like accuracy are translated to a loss.
print(run_value)

############################################################################
# Fit an user provided pipeline
# =============================

# We will create a configuration that has a user defined
# min_samples_split in the Random Forest. We recommend you to look into
# how the ConfigSpace package works here:
# https://automl.github.io/ConfigSpace/master/
cs = cls.get_configuration_space(X, y)
config = cs.sample_configuration()
print("Configuration before the change:", config)
config_dict = config.get_dictionary()
config_dict['classifier:random_forest:min_samples_split'] = 11
config = Configuration(cs, config_dict)
print("Configuration after the change:", config)

# Fit the configuration
pipeline = cls.fit_pipeline(X=X_train, y=y_train, config=config)[0]

# We can make sure that our pipeline configuration was honored as follows
print("Passed Configuration:", pipeline.config)
print("Random Forest:", pipeline.named_steps['classifier'].choice.estimator)
