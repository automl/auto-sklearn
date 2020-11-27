# -*- encoding: utf-8 -*-
"""
===========================================
Parallel Usage with manual process spawning
===========================================

*Auto-sklearn* uses
`dask.distributed <https://distributed.dask.org/en/latest/index.html`>_
for parallel optimization.

This example shows how to spawn workers for *Auto-sklearn* manually.
Use this example as a starting point to parallelize *Auto-sklearn*
across multiple machines. To run *Auto-sklearn* in parallel
on a single machine check out the example
`Parallel Usage on a single machine <example_parallel_n_jobs.html>`_.
"""

import asyncio
import multiprocessing
import subprocess
import time

import dask
import dask.distributed
import sklearn.datasets
import sklearn.metrics

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import MULTICLASS_CLASSIFICATION

tmp_folder = '/tmp/autosklearn_parallel_2_example_tmp'
output_folder = '/tmp/autosklearn_parallel_2_example_out'


############################################################################
# Dask configuration
# ==================
#
# Auto-sklearn uses threads in Dask to launch memory constrained jobs.
# This number of threads can be provided directly via the n_jobs argument
# when creating the AutoSklearnClassifier. Additionally, the user can provide
# a dask_client argument which can have processes=True.
# When using processes to True, we need to specify the below setting
# to allow internally generated processes.
# Optionally, you can choose to provide a dask client with processes=False
# and remove the following line.
dask.config.set({'distributed.worker.daemon': False})


############################################################################
# Start worker - Python
# =====================
#
# This function demonstrates how to start a dask worker from python. This
# is a bit cumbersome and should ideally be done from the command line.
# We do it here for illustrational purpose, butalso start one worker from
# the command line below.

# Check the dask docs at
# https://docs.dask.org/en/latest/setup/python-advanced.html for further
# information.


def start_python_worker(scheduler_address):
    dask.config.set({'distributed.worker.daemon': False})

    async def do_work():
        async with dask.distributed.Nanny(
                scheduler_ip=scheduler_address,
                nthreads=1,
                lifetime=35,  # automatically shut down the worker so this loop ends
        ) as worker:
            await worker.finished()

    asyncio.get_event_loop().run_until_complete(do_work())


############################################################################
# Start worker - CLI
# ==================
#
# It is also possible to start dask workers from the command line (in fact,
# one can also start a dask scheduler from the command line), see the
# `dask cli docs <https://docs.dask.org/en/latest/setup/cli.html>`_ for
# further information.
# Please not, that DASK_DISTRIBUTED__WORKER__DAEMON=False is required in this
# case as dask-worker creates a new process. That is, it is equivalent to the
# setting described above with dask.distributed.Client with processes=True
#
# Again, we need to make sure that we do not start the workers in a daemon
# mode.

def start_cli_worker(scheduler_address):
    call_string = (
        "DASK_DISTRIBUTED__WORKER__DAEMON=False "
        "dask-worker %s --nthreads 1 --lifetime 35"
    ) % scheduler_address
    proc = subprocess.run(call_string, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=True, check=True)
    while proc.returncode is None:
        time.sleep(1)


############################################################################
# Start Auto-sklearn
# ==================
#
# We are now ready to start *auto-sklearn.
#
# To use auto-sklearn in parallel we must guard the code with
# ``if __name__ == '__main__'``. We then start a dask cluster as a context,
# which means that it is automatically stopped one all computation is done.
if __name__ == '__main__':
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Create a dask compute cluster and a client manually - the former can
    # be done via command line, too.
    with dask.distributed.LocalCluster(
        n_workers=0, processes=True, threads_per_worker=1,
    ) as cluster, dask.distributed.Client(address=cluster.scheduler_address) as client:

        # now we start the two workers, one from within Python, the other
        # via the command line.
        process_python_worker = multiprocessing.Process(
            target=start_python_worker,
            args=(cluster.scheduler_address,),
        )
        process_python_worker.start()
        process_cli_worker = multiprocessing.Process(
            target=start_cli_worker,
            args=(cluster.scheduler_address,),
        )
        process_cli_worker.start()

        # Wait a second for workers to become available
        time.sleep(1)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            memory_limit=1024,
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            seed=777,
            # n_jobs is ignored internally as we pass a dask client.
            n_jobs=1,
            # Pass a dask client which connects to the previously constructed cluster.
            dask_client=client,
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

        # Wait until all workers are closed
        process_python_worker.join()
        process_cli_worker.join()
