# -*- encoding: utf-8 -*-
"""
======================================================
Parallel Usage: Spawning workers from the command line
======================================================

*Auto-sklearn* uses
`dask.distributed <https://distributed.dask.org/en/latest/index.html>`_
for parallel optimization.

This example shows how to start the dask scheduler and spawn
workers for *Auto-sklearn* manually from the command line. Use this example
as a starting point to parallelize *Auto-sklearn* across multiple
machines. If you want to start everything manually from within Python
please see `this example <example_parallel_manual_spawning_python.html>`_.
To run *Auto-sklearn* in parallel on a single machine check out the example
`Parallel Usage on a single machine <example_parallel_n_jobs.html>`_.

You can learn more about the dask command line interface from
https://docs.dask.org/en/latest/setup/cli.html.

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

###########################################################################
# Import statements
# =================

import multiprocessing
import subprocess
import time

import dask.distributed
import sklearn.datasets
import sklearn.metrics

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import MULTICLASS_CLASSIFICATION

tmp_folder = '/tmp/autosklearn_parallel_3_example_tmp'
output_folder = '/tmp/autosklearn_parallel_3_example_out'

worker_processes = []


###########################################################################
# 0. Setup client-scheduler communication
# =======================================
#
# In this examples the dask scheduler is started without an explicit
# address and port. Instead, the scheduler takes a free port and stores
# relevant information in a file for which we provided the name and
# location. This filename is also given to the worker so they can find all
# relevant information to connect to the scheduler.

scheduler_file_name = 'scheduler-file.json'


############################################################################
# 1. Start scheduler
# ==================
#
# Starting the scheduler is done with the following bash command:
#
# .. code:: bash
#
#     dask-scheduler --scheduler-file scheduler-file.json --idle-timeout 10
#
# We will now execute this bash command from within Python to have a
# self-contained example:

def cli_start_scheduler(scheduler_file_name):
    call_string = (
        "dask-scheduler --scheduler-file %s --idle-timeout 10"
    ) % scheduler_file_name
    proc = subprocess.run(call_string, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, shell=True, check=True)
    while proc.returncode is None:
        time.sleep(1)


if __name__ == "__main__":
    process_python_worker = multiprocessing.Process(
        target=cli_start_scheduler,
        args=(scheduler_file_name, ),
    )
    process_python_worker.start()
    worker_processes.append(process_python_worker)

    # Wait a second for the scheduler to become available
    time.sleep(1)


############################################################################
# 2. Start two workers
# ====================
#
# Starting the scheduler is done with the following bash command:
#
# .. code:: bash
#
#     DASK_DISTRIBUTED__WORKER__DAEMON=False \
#         dask-worker --nthreads 1 --lifetime 35 --memory-limit 0 \
#         --scheduler-file scheduler-file.json
#
# We will now execute this bash command from within Python to have a
# self-contained example. Please note, that
# ``DASK_DISTRIBUTED__WORKER__DAEMON=False`` is required in this
# case as dask-worker creates a new process, which by default is not
# compatible with Auto-sklearn creating new processes in the workers itself.
# We disable dask's memory management by passing ``--memory-limit`` as
# Auto-sklearn does the memory management itself.

def cli_start_worker(scheduler_file_name):
    call_string = (
        "DASK_DISTRIBUTED__WORKER__DAEMON=False "
        "dask-worker --nthreads 1 --lifetime 35 --memory-limit 0 "
        "--scheduler-file %s"
    ) % scheduler_file_name
    proc = subprocess.run(call_string, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, shell=True)
    while proc.returncode is None:
        time.sleep(1)

if __name__ == '__main__':
    for _ in range(2):
        process_cli_worker = multiprocessing.Process(
            target=cli_start_worker,
            args=(scheduler_file_name, ),
        )
        process_cli_worker.start()
        worker_processes.append(process_cli_worker)

    # Wait a second for workers to become available
    time.sleep(1)

############################################################################
# 3. Creating a client in Python
# ==============================
#
# Finally we create a dask cluster which also connects to the scheduler via
# the information in the file created by the scheduler.

client = dask.distributed.Client(scheduler_file=scheduler_file_name)

############################################################################
# Start Auto-sklearn
# ~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
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


############################################################################
# Wait until all workers are closed
# =================================
#
# This is only necessary if the workers are started from within this python
# script. In a real application one would start them directly from the command
# line.
if __name__ == '__main__':
    process_python_worker.join()
    for process in worker_processes:
        process.join()
