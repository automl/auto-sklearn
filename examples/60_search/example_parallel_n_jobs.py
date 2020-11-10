# -*- encoding: utf-8 -*-
"""
===================================
Parallel Usage  on a single machine
===================================

*Auto-sklearn* uses
`dask.distributed <https://distributed.dask.org/en/latest/index.html`>_
for parallel optimization.

This example shows how to start *Auto-sklearn* to use multiple cores on a
single machine. Using this mode, *Auto-sklearn* starts a dask cluster,
manages the workers and takes care of shutting down the cluster once the
computation is done.
To run *Auto-sklearn* on multiple machines check the example
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
#
# To use ``n_jobs_`` we must guard the code
if __name__ == '__main__':

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_parallel_1_example_tmp',
        output_folder='/tmp/autosklearn_parallel_1_example_out',
        n_jobs=4,
        # Each one of the 4 jobs is allocated 3GB
        memory_limit=3072,
        seed=5,
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
