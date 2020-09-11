# -*- encoding: utf-8 -*-
"""
===========================================
Parallel Usage with manual process spawning
===========================================

*Auto-sklearn* uses *SMAC* to automatically optimize the hyperparameters of
the training models. A variant of *SMAC*, called *pSMAC* (parallel SMAC),
provides a means of running several instances of *auto-sklearn* in a parallel
mode using several computational resources (detailed information of *pSMAC*
can be found `here <https://automl.github.io/SMAC3/master/psmac.html>`_).

This example shows how to spawn workers for *Auto-sklearn* manually.
Use this example as a starting point to parallelize *Auto-sklearn*
across multiple machines. To run *Auto-sklearn* in parallel
on a single machine check out the example
`Parallel Usage on a single machine <example_parallel_n_jobs.html>`_.
"""

import json
import multiprocessing
import os
import time

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import MULTICLASS_CLASSIFICATION

import dask
from dask.distributed import Nanny

tmp_folder = '/tmp/autosklearn_parallel_2_example_tmp'
output_folder = '/tmp/autosklearn_parallel_2_example_out'

############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Define helper functions
# =======================
#
# For this example we need to start Auto-sklearn in a separate process
# first. You can think of this as starting Auto-sklearn on machine 1.
# Auto-sklearn then spawns a dask server locally. Next, we create
# a worker and connect to that server in the main process. You can think
# of this as starting the worker on machine 2. All further processing
# like predicting needs to be done in the process running Auto-sklearn.

def run_autosklearn(X_train, y_train, X_test, y_test, tmp_folder, output_folder):
    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        ml_memory_limit=1024,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        seed=777,
        # n_jobs needs to be set to 1
        n_jobs=1,
    )
    automl.fit(X_train, y_train)

    automl.fit_ensemble(
        y_train,
        task=MULTICLASS_CLASSIFICATION,
        dataset_name='digits',
        ensemble_size=20,
        ensemble_nbest=50,
    )

    predictions = automl.predict(X_test)
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

############################################################################
# Start Auto-sklearn on "Machine 1"
# =================================
#
# To use ``n_jobs_`` we must guard the code
if __name__ == '__main__':
    process = multiprocessing.Process(
        target=run_autosklearn,
        args=(X_train, y_train, X_test, y_test, tmp_folder, output_folder),
    )
    process.start()

############################################################################
# Start a worker on "Machine 2"
# =============================
#
# To do so, we first need to obtain information on how to connect to the
# server. This is stored in a subdirectory of the temporary directory given
# above.

    dask_scheduler_file = os.path.join(tmp_folder, 'smac3-output', 'run_777', '.dask_scheduler_file')
    # The "777" is the seed given above
    while True:
        try:
            with open(dask_scheduler_file) as fh:
                dask_configuration = json.load(fh)
            break
        except:
            time.sleep(0.1)

    # Auto-sklearn requires dask workers to not run in the daemon setting
    dask.config.set({'distributed.worker.daemon': False})

    # Starting a dask worker in python is a bit cumbersome and should ideally
    # be done from the command line (we do it here only to keep the example
    # to a single script). Check the dask docs at
    # https://docs.dask.org/en/latest/setup/python-advanced.html for further
    # information
    import asyncio
    async def do_work():
        async with Nanny(scheduler_file=dask_scheduler_file, nthreads=1, ncores=3) as worker:
            await worker.finished()

    asyncio.get_event_loop().run_until_complete(do_work())

    process.join()
