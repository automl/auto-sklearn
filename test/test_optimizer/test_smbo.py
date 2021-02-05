import logging.handlers

from ConfigSpace.configuration_space import Configuration

import pytest

import autosklearn.metrics
from autosklearn.smbo import AutoMLSMBO
import autosklearn.pipeline.util as putil
from autosklearn.automl import AutoML
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.util.stopwatch import StopWatch


@pytest.mark.parametrize("context", ['fork', 'forkserver'])
def test_smbo_metalearning_configurations(backend, context, dask_client):

    # Get the inputs to the optimizer
    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    config_space = AutoML(backend=backend,
                          metric=autosklearn.metrics.accuracy,
                          time_left_for_this_task=20,
                          per_run_time_limit=5).fit(
                              X_train, Y_train,
                              task=BINARY_CLASSIFICATION,
                              only_return_configuration_space=True)
    watcher = StopWatch()

    # Create an optimizer
    smbo = AutoMLSMBO(
        config_space=config_space,
        dataset_name='iris',
        backend=backend,
        total_walltime_limit=10,
        func_eval_time_limit=5,
        memory_limit=4096,
        metric=autosklearn.metrics.accuracy,
        watcher=watcher,
        n_jobs=1,
        dask_client=dask_client,
        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        start_num_run=1,
        data_memory_limit=None,
        num_metalearning_cfgs=25,
        pynisher_context=context,
    )
    assert smbo.pynisher_context == context

    # Create the inputs to metalearning
    datamanager = XYDataManager(
        X_train, Y_train,
        X_test, Y_test,
        task=BINARY_CLASSIFICATION,
        dataset_name='iris',
        feat_type=None,
    )
    backend.save_datamanager(datamanager)
    smbo.task = BINARY_CLASSIFICATION
    smbo.reset_data_manager()
    metalearning_configurations = smbo.get_metalearning_suggestions()

    # We should have 25 metalearning configurations
    assert len(metalearning_configurations) == 25
    assert [isinstance(config, Configuration) for config in metalearning_configurations]
