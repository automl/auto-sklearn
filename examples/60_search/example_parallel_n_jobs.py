# -*- encoding: utf-8 -*-
"""
===================================
Parallel Usage  on a single machine
===================================

*Auto-sklearn* uses *SMAC* to automatically optimize the hyperparameters of
the training models. A variant of *SMAC*, called *pSMAC* (parallel SMAC),
provides a means of running several instances of *auto-sklearn* in a parallel
mode using several computational resources (detailed information of *pSMAC*
can be found `here <https://automl.github.io/SMAC3/master/psmac.html>`_).

This example shows how to start *Auto-sklearn* to use multiple cores on a
single machine. To run *Auto-sklearn* on multiple machines check the example
`Parallel Usage with manual process spawning <example_parallel_manual_spawning.html>`_.
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
# Build and fit a classifier
# ==========================
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
    output_folder='/tmp/autosklearn_parallel_1_example_out',
    disable_evaluator_output=False,
    # 'holdout' with 'train_size'=0.67 is the default argument setting
    # for AutoSklearnClassifier. It is explicitly specified in this example
    # for demonstrational purpose.
    resampling_strategy='holdout',
    resampling_strategy_arguments={'train_size': 0.67},
    n_jobs=4,
    # Each one of the 4 jobs is allocated 3GB
    ml_memory_limit=3072,
    seed=5,
    delete_output_folder_after_terminate=False,
    delete_tmp_folder_after_terminate=False,
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================
print(automl.show_models())

############################################################################
# Print statistics about the auto-sklearn run
# ===========================================

# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())

############################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
