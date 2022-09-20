# -*- encoding: utf-8 -*-
"""
=====================
Logging and debugging
=====================

This example shows how to provide a custom logging configuration to *auto-sklearn*.
We will be fitting 2 pipelines and showing any INFO-level msg on console.
Even if you do not provide a logging_configuration, autosklearn creates a log file
in the temporal working directory. This directory can be specified via the `tmp_folder`
as exemplified below.

This example also highlights additional information about *auto-sklearn* internal
directory structure.
"""
import pathlib

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

import autosklearn.classification


############################################################################
# Data Loading
# ============
# Load kr-vs-kp dataset from https://www.openml.org/d/3
X, y = data = sklearn.datasets.fetch_openml(data_id=3, return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)


############################################################################
# Create a logging config
# =======================
# *auto-sklearn* uses a default
# `logging config <https://github.com/automl/auto-sklearn/blob/master/autosklearn/util/logging.yaml>`_
# We will instead create a custom one as follows:

logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "custom": {
            # More format options are available in the official
            # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    # Any INFO level msg will be printed to the console
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "custom",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "DEBUG",
        },
        "Client-EnsembleBuilder": {
            "level": "DEBUG",
            "handlers": ["console"],
        },
    },
}


############################################################################
# Build and fit a classifier
# ==========================
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={"runcount_limit": 2},
    # Pass the config file we created
    logging_config=logging_config,
    # *auto-sklearn* generates temporal files under tmp_folder
    tmp_folder="./tmp_folder",
    # By default tmp_folder is deleted. We will preserve it
    # for debug purposes
    delete_tmp_folder_after_terminate=False,
)
cls.fit(X_train, y_train, X_test, y_test)

# *auto-sklearn* generates intermediate files which can be of interest
# Dask multiprocessing information. Useful on multi-core runs:
#   * tmp_folder/distributed.log
# The individual fitted estimators are written to disk on:
#   * tmp_folder/.auto-sklearn/runs
# SMAC output is stored in this directory.
# For more info, you can check the `SMAC documentation <https://github.com/automl/SMAC3>`_
#   * tmp_folder/smac3-output
# Auto-sklearn always outputs to this log file
# tmp_folder/AutoML*.log
for filename in pathlib.Path("./tmp_folder").glob("*"):
    print(filename)
