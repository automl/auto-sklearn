# -*- encoding: utf-8 -*-
"""
===================================================
Parallel Usage: Spawning workers from within Python
===================================================

*Auto-sklearn* uses
`dask.distributed <https://distributed.dask.org/en/latest/index.html>`_
for parallel optimization.

This example shows how to start the dask scheduler and spawn
workers for *Auto-sklearn* manually within Python. Use this example
as a starting point to parallelize *Auto-sklearn* across multiple
machines. If you want to start everything manually from the command line
please see `this example <example_parallel_manual_spawning_cli.html>`_.
To run *Auto-sklearn* in parallel on a single machine check out the example
`Parallel Usage on a single machine <example_parallel_n_jobs.html>`_.

When manually passing a dask client to Auto-sklearn, all logic
must be guarded by ``if __name__ == "__main__":`` statements! We use
multiple such statements to properly render this example as a notebook
and also allow execution via the command line.

Background
==========

To run Auto-sklearn distributed on multiple machines we need to set
up three components:

1. **Auto-sklearn and a dask client**. This will manage all workload, find new
   configurations to evaluate and submit jobs via a dask client. As this
   runs Bayesian optimization it should be executed on its own CPU.
2. **The dask workers**. They will do the actual work of running machine
   learning algorithms and require their own CPU each.
3. **The scheduler**. It manages the communication between the dask client
   and the different dask workers. As the client and all workers connect
   to the scheduler it must be started first. This is a light-weight job
   and does not require its own CPU.

We will now start these three components in reverse order: scheduler,
workers and client. Also, in a real setup, the scheduler and the workers should
be started from the command line and not from within a Python file via
the ``subprocess`` module as done here (for the sake of having a self-contained
example).
"""

import asyncio
import multiprocessing
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
# Define function to start worker
# ===============================
#
# Define the function to start a dask worker from python. This
# is a bit cumbersome and should ideally be done from the command line.
# We do it here only for illustrational purpose.

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
            memory_limit=0,  # Disable memory management as it is done by Auto-sklearn itself
        ) as worker:
            await worker.finished()

    asyncio.get_event_loop().run_until_complete(do_work())


############################################################################
# Start Auto-sklearn
# ==================
#
# We are now ready to start *auto-sklearn and all dask related processes.
#
# To use auto-sklearn in parallel we must guard the code with
# ``if __name__ == '__main__'``. We then start a dask cluster as a context,
# which means that it is automatically stopped once all computation is done.
if __name__ == '__main__':
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    # 1. Create a dask scheduler (LocalCluster)
    with dask.distributed.LocalCluster(
        n_workers=0, processes=True, threads_per_worker=1,
    ) as cluster:

        # 2. Start the workers
        # now we start the two workers, one from within Python, the other
        # via the command line.
        worker_processes = []
        for _ in range(2):
            process_python_worker = multiprocessing.Process(
                target=start_python_worker,
                args=(cluster.scheduler_address, ),
            )
            process_python_worker.start()
            worker_processes.append(process_python_worker)

        # Wait a second for workers to become available
        time.sleep(1)

        # 3. Start the client
        with dask.distributed.Client(address=cluster.scheduler_address) as client:
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
        for process in worker_processes:
            process_python_worker.join()
