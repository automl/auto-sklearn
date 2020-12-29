# -*- encoding: utf-8 -*-
import itertools
import os
import pickle
import sys
import time
import glob
import unittest
import unittest.mock

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
from smac.scenario.scenario import Scenario
from smac.facade.roar_facade import ROAR

from autosklearn.automl import AutoML
import autosklearn.automl
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.metrics import accuracy, log_loss, balanced_accuracy
import autosklearn.pipeline.util as putil
from autosklearn.constants import (
    MULTICLASS_CLASSIFICATION,
    BINARY_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    REGRESSION,
    MULTIOUTPUT_REGRESSION,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
)
from smac.tae import StatusType

sys.path.append(os.path.dirname(__file__))
from automl_utils import print_debug_information, count_succeses, AutoMLLogParser  # noqa (E402: module level import not at top of file)


class AutoMLStub(AutoML):
    def __init__(self):
        self.__class__ = AutoML
        self._task = None
        self._dask_client = None
        self._is_dask_client_internally_created = False

    def __del__(self):
        pass


def test_fit(dask_client, backend):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        backend=backend,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        metric=accuracy,
        dask_client=dask_client,
    )
    automl.fit(
        X_train, Y_train, task=MULTICLASS_CLASSIFICATION,
    )
    score = automl.score(X_test, Y_test)
    assert score > 0.8
    assert count_succeses(automl.cv_results_) > 0
    assert automl._task == MULTICLASS_CLASSIFICATION

    del automl


def test_fit_roar(dask_client_single_worker, backend):
    def get_roar_object_callback(
            scenario_dict,
            seed,
            ta,
            ta_kwargs,
            dask_client,
            n_jobs,
            **kwargs
    ):
        """Random online adaptive racing.

        http://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf"""
        scenario = Scenario(scenario_dict)
        return ROAR(
            scenario=scenario,
            rng=seed,
            tae_runner=ta,
            tae_runner_kwargs=ta_kwargs,
            dask_client=dask_client,
            n_jobs=n_jobs,
        )

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        backend=backend,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        initial_configurations_via_metalearning=0,
        get_smac_object_callback=get_roar_object_callback,
        metric=accuracy,
        dask_client=dask_client_single_worker,
    )
    automl.fit(
        X_train, Y_train, task=MULTICLASS_CLASSIFICATION,
    )
    score = automl.score(X_test, Y_test)
    assert score > 0.8
    assert count_succeses(automl.cv_results_) > 0
    assert automl._task == MULTICLASS_CLASSIFICATION

    del automl


def test_refit_shuffle_on_fail(backend, dask_client):

    failing_model = unittest.mock.Mock()
    failing_model.fit.side_effect = [ValueError(), ValueError(), None]
    failing_model.fit_transformer.side_effect = [
        ValueError(), ValueError(), (None, {})]
    failing_model.get_max_iter.return_value = 100

    auto = AutoML(backend, 30, 5, dask_client=dask_client)
    ensemble_mock = unittest.mock.Mock()
    ensemble_mock.get_selected_model_identifiers.return_value = [(1, 1, 50.0)]
    auto.ensemble_ = ensemble_mock
    for budget_type in [None, 'iterations']:
        auto._budget_type = budget_type

        auto.models_ = {(1, 1, 50.0): failing_model}

        # Make sure a valid 2D array is given to automl
        X = np.array([1, 2, 3]).reshape(-1, 1)
        y = np.array([1, 2, 3])
        auto.refit(X, y)

        assert failing_model.fit.call_count == 3
    assert failing_model.fit_transformer.call_count == 3

    del auto


def test_only_loads_ensemble_models(automl_stub):

    def side_effect(ids, *args, **kwargs):
        return models if ids is identifiers else {}

    # Add a resampling strategy as this is required by load_models
    automl_stub._resampling_strategy = 'holdout'
    identifiers = [(1, 2), (3, 4)]

    models = [42]
    load_ensemble_mock = unittest.mock.Mock()
    load_ensemble_mock.get_selected_model_identifiers.return_value = identifiers
    automl_stub._backend.load_ensemble.return_value = load_ensemble_mock
    automl_stub._backend.load_models_by_identifiers.side_effect = side_effect

    automl_stub._load_models()
    assert models == automl_stub.models_
    assert automl_stub.cv_models_ is None

    automl_stub._resampling_strategy = 'cv'

    models = [42]
    automl_stub._backend.load_cv_models_by_identifiers.side_effect = side_effect

    automl_stub._load_models()
    assert models == automl_stub.cv_models_


def test_check_for_models_if_no_ensemble(automl_stub):
    models = [42]
    automl_stub._backend.load_ensemble.return_value = None
    automl_stub._backend.list_all_models.return_value = models
    automl_stub._disable_evaluator_output = False

    automl_stub._load_models()


def test_raises_if_no_models(automl_stub):
    automl_stub._backend.load_ensemble.return_value = None
    automl_stub._backend.list_all_models.return_value = []
    automl_stub._resampling_strategy = 'holdout'

    automl_stub._disable_evaluator_output = False
    with pytest.raises(ValueError):
        automl_stub._load_models()

    automl_stub._disable_evaluator_output = True
    automl_stub._load_models()


def test_delete_non_candidate_models(backend, dask_client):

    seed = 555
    X, Y, _, _ = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        backend,
        time_left_for_this_task=60,
        per_run_time_limit=5,
        ensemble_nbest=3,
        seed=seed,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        include_estimators=['sgd'],
        include_preprocessors=['no_preprocessing'],
        metric=accuracy,
        dask_client=dask_client,
        # Force model to be deleted. That is, from 50 which is the
        # default to 3 to make sure we delete models.
        max_models_on_disc=3,
    )

    automl.fit(X, Y, task=MULTICLASS_CLASSIFICATION,
               X_test=X, y_test=Y)

    # Assert at least one model file has been deleted and that there were no
    # deletion errors
    log_file_path = glob.glob(os.path.join(
        backend.temporary_directory, 'AutoML(' + str(seed) + '):*.log'))
    with open(log_file_path[0]) as log_file:
        log_content = log_file.read()
        assert 'Deleted files of non-candidate model' in log_content, log_content
        assert 'Failed to delete files of non-candidate model' not in log_content, log_content
        assert 'Failed to lock model' not in log_content, log_content

    # Assert that the files of the models used by the ensemble weren't deleted
    model_files = backend.list_all_models(seed=seed)
    model_files_idx = set()
    for m_file in model_files:
        # Extract the model identifiers from the filename
        m_file = os.path.split(m_file)[1].replace('.model', '').split('.', 2)
        model_files_idx.add((int(m_file[0]), int(m_file[1]), float(m_file[2])))
    ensemble_members_idx = set(automl.ensemble_.identifiers_)
    assert ensemble_members_idx.issubset(model_files_idx), (ensemble_members_idx, model_files_idx)

    del automl


def test_binary_score_and_include(backend, dask_client):
    """
    Test fix for binary classification prediction
    taking the index 1 of second dimension in prediction matrix
    """

    data = sklearn.datasets.make_classification(
        n_samples=400, n_features=10, n_redundant=1, n_informative=3,
        n_repeated=1, n_clusters_per_class=2, random_state=1)
    X_train = data[0][:200]
    Y_train = data[1][:200]
    X_test = data[0][200:]
    Y_test = data[1][200:]

    automl = autosklearn.automl.AutoML(
        backend, 20, 5,
        include_estimators=['sgd'],
        include_preprocessors=['no_preprocessing'],
        metric=accuracy,
        dask_client=dask_client,
    )
    automl.fit(X_train, Y_train, task=BINARY_CLASSIFICATION)
    assert automl._task == BINARY_CLASSIFICATION

    # TODO, the assumption from above is not really tested here
    # Also, the score method should be removed, it only makes little sense
    score = automl.score(X_test, Y_test)
    assert score >= 0.4

    del automl


def test_automl_outputs(backend, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    name = 'iris'
    data_manager_file = os.path.join(
        backend.temporary_directory,
        '.auto-sklearn',
        'datamanager.pkl'
    )

    auto = autosklearn.automl.AutoML(
        backend, 30, 5,
        initial_configurations_via_metalearning=0,
        seed=100,
        metric=accuracy,
        dask_client=dask_client,
    )
    auto.fit(
        X=X_train,
        y=Y_train,
        X_test=X_test,
        y_test=Y_test,
        dataset_name=name,
        task=MULTICLASS_CLASSIFICATION,
    )

    # pickled data manager (without one hot encoding!)
    with open(data_manager_file, 'rb') as fh:
        D = pickle.load(fh)
        assert np.allclose(D.data['X_train'], X_train)

    # Check that all directories are there
    fixture = [
        'true_targets_ensemble.npy',
        'start_time_100',
        'datamanager.pkl',
        'ensemble_read_preds.pkl',
        'ensemble_read_scores.pkl',
        'runs',
        'ensembles',
        'ensemble_history.json',
    ]
    assert (
        sorted(os.listdir(os.path.join(backend.temporary_directory, '.auto-sklearn')))
        == sorted(fixture)
    )

    # At least one ensemble, one validation, one test prediction and one
    # model and one ensemble
    fixture = glob.glob(os.path.join(
        backend.temporary_directory,
        '.auto-sklearn', 'runs', '*', 'predictions_ensemble*npy',
    ))
    assert len(fixture) > 0

    fixture = glob.glob(os.path.join(backend.temporary_directory, '.auto-sklearn',
                                     'runs', '*', '100.*.model'))
    assert len(fixture) > 0

    fixture = os.listdir(os.path.join(backend.temporary_directory,
                                      '.auto-sklearn', 'ensembles'))
    assert '100.0000000000.ensemble' in fixture

    # Start time
    start_time_file_path = os.path.join(backend.temporary_directory,
                                        '.auto-sklearn', "start_time_100")
    with open(start_time_file_path, 'r') as fh:
        start_time = float(fh.read())
    assert time.time() - start_time >= 10, print_debug_information(auto)

    # Then check that the logger matches the run expectation
    logfile = glob.glob(os.path.join(
           auto._backend.temporary_directory, 'AutoML*.log'))[0]
    parser = AutoMLLogParser(logfile)

    # The number of ensemble trajectories properly in log file
    success_ensemble_iters_auto = len(auto.ensemble_performance_history)
    success_ensemble_iters_log = parser.count_ensembler_success_pynisher_calls()
    assert success_ensemble_iters_auto == success_ensemble_iters_log, "{} != {}".format(
        auto.ensemble_performance_history,
        print_debug_information(auto),
    )

    # We also care that no iteration got lost
    # This is important because it counts for pynisher calls
    # and whether a pynisher call actually called the ensemble
    total_ensemble_iterations = parser.count_ensembler_iterations()
    assert len(total_ensemble_iterations) > 1  # At least 1 iteration
    assert range(1, max(total_ensemble_iterations) + 1), total_ensemble_iterations

    # a point where pynisher is called before budget exhaustion
    # Dummy not in run history
    total_calls_to_pynisher_log = parser.count_tae_pynisher_calls() - 1
    total_returns_from_pynisher_log = parser.count_tae_pynisher_returns() - 1
    total_elements_rh = len([run_value for run_value in auto.runhistory_.data.values(
    ) if run_value.status == StatusType.RUNNING])

    # Make sure we register all calls to pynisher
    # The less than or equal here is added as a WA as
    # https://github.com/automl/SMAC3/pull/712 is not yet integrated
    assert total_elements_rh <= total_calls_to_pynisher_log, print_debug_information(auto)

    # Make sure we register all returns from pynisher
    assert total_elements_rh <= total_returns_from_pynisher_log, print_debug_information(auto)

    # Lastly check that settings are print to logfile
    ensemble_size = parser.get_automl_setting_from_log(auto._dataset_name, 'ensemble_size')
    assert auto._ensemble_size == int(ensemble_size)

    del auto


@pytest.mark.parametrize("datasets", [('breast_cancer', BINARY_CLASSIFICATION),
                                      ('wine', MULTICLASS_CLASSIFICATION),
                                      ('diabetes', REGRESSION)])
def test_do_dummy_prediction(backend, dask_client, datasets):

    name, task = datasets

    X_train, Y_train, X_test, Y_test = putil.get_dataset(name)
    datamanager = XYDataManager(
        X_train, Y_train,
        X_test, Y_test,
        task=task,
        dataset_name=name,
        feat_type=None,
    )

    auto = autosklearn.automl.AutoML(
        backend, 20, 5,
        initial_configurations_via_metalearning=25,
        metric=accuracy,
        dask_client=dask_client,
    )

    # Make a dummy logger
    auto._logger_port = 9020
    auto._logger = unittest.mock.Mock()
    auto._logger.info.return_value = None

    auto._backend.save_datamanager(datamanager)
    D = backend.load_datamanager()

    # Check if data manager is correcly loaded
    assert D.info['task'] == datamanager.info['task']
    auto._do_dummy_prediction(D, 1)

    # Ensure that the dummy predictions are not in the current working
    # directory, but in the temporary directory.
    assert not os.path.exists(os.path.join(os.getcwd(), '.auto-sklearn'))
    assert os.path.exists(os.path.join(
        backend.temporary_directory, '.auto-sklearn', 'runs', '1_1_0.0',
        'predictions_ensemble_1_1_0.0.npy')
    )

    auto._clean_logger()

    del auto


@unittest.mock.patch('autosklearn.evaluation.ExecuteTaFuncWithQueue.run')
def test_fail_if_dummy_prediction_fails(ta_run_mock, backend, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    datamanager = XYDataManager(
        X_train, Y_train,
        X_test, Y_test,
        task=2,
        feat_type=['Numerical' for i in range(X_train.shape[1])],
        dataset_name='iris',
    )

    time_for_this_task = 30
    per_run_time = 10
    auto = autosklearn.automl.AutoML(backend,
                                     time_for_this_task,
                                     per_run_time,
                                     initial_configurations_via_metalearning=25,
                                     metric=accuracy,
                                     dask_client=dask_client,
                                     )
    auto._backend._make_internals_directory()
    auto._backend.save_datamanager(datamanager)

    # Make a dummy logger
    auto._logger_port = 9020
    auto._logger = unittest.mock.Mock()
    auto._logger.info.return_value = None

    # First of all, check that ta.run() is actually called.
    ta_run_mock.return_value = StatusType.SUCCESS, None, None, {}
    auto._do_dummy_prediction(datamanager, 1)
    ta_run_mock.assert_called_once_with(1, cutoff=time_for_this_task)

    # Case 1. Check that function raises no error when statustype == success.
    # ta.run() returns status, cost, runtime, and additional info.
    ta_run_mock.return_value = StatusType.SUCCESS, None, None, {}
    raised = False
    try:
        auto._do_dummy_prediction(datamanager, 1)
    except ValueError:
        raised = True
    assert not raised, 'Exception raised'

    # Case 2. Check that if statustype returned by ta.run() != success,
    # the function raises error.
    ta_run_mock.return_value = StatusType.CRASHED, None, None, {}
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.CRASHED and additional output: {}.'  # noqa
    ):
        auto._do_dummy_prediction(datamanager, 1)

    ta_run_mock.return_value = StatusType.ABORT, None, None, {}
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.ABORT '
              'and additional output: {}.',
    ):
        auto._do_dummy_prediction(datamanager, 1)
    ta_run_mock.return_value = StatusType.TIMEOUT, None, None, {}
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.TIMEOUT '
              'and additional output: {}.'
    ):
        auto._do_dummy_prediction(datamanager, 1)
    ta_run_mock.return_value = StatusType.MEMOUT, None, None, {}
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.MEMOUT '
              'and additional output: {}.',
    ):
        auto._do_dummy_prediction(datamanager, 1)
    ta_run_mock.return_value = StatusType.CAPPED, None, None, {}
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.CAPPED '
              'and additional output: {}.'
    ):
        auto._do_dummy_prediction(datamanager, 1)

    ta_run_mock.return_value = StatusType.CRASHED, None, None, {'exitcode': -6}
    with pytest.raises(
        ValueError,
        match='The error suggests that the provided memory limits were too tight.',
    ):
        auto._do_dummy_prediction(datamanager, 1)


@unittest.mock.patch('autosklearn.smbo.AutoMLSMBO.run_smbo')
def test_exceptions_inside_log_in_smbo(smbo_run_mock, backend, dask_client):

    # Below importing and shutdown is a workaround, to make sure
    # we reset the port to collect messages. Randomly, when running
    # this test with multiple other test at the same time causes this
    # test to fail. This resets the singletons of the logging class
    import logging
    logging.shutdown()

    automl = autosklearn.automl.AutoML(
        backend,
        20,
        5,
        metric=accuracy,
        dask_client=dask_client,
    )

    dataset_name = 'test_exceptions_inside_log'

    # Create a custom exception to prevent other errors to slip in
    class MyException(Exception):
        pass

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    # The first call is on dummy predictor failure
    message = str(np.random.randint(100)) + '_run_smbo'
    smbo_run_mock.side_effect = MyException(message)

    with pytest.raises(MyException):
        automl.fit(
            X_train,
            Y_train,
            task=MULTICLASS_CLASSIFICATION,
            dataset_name=dataset_name,
        )

    # make sure that the logfile was created
    import shutil
    shutil.copytree(backend.temporary_directory, '/tmp/trydebug')
    logger_name = 'AutoML(%d):%s' % (1, dataset_name)
    logfile = os.path.join(backend.temporary_directory, logger_name + '.log')
    assert os.path.exists(logfile), automl._clean_logger()
    with open(logfile) as f:
        assert message in f.read(), automl._clean_logger()

    # Speed up the closing after forced crash
    automl._clean_logger()


@pytest.mark.parametrize("metric", [log_loss, balanced_accuracy])
def test_load_best_individual_model(metric, backend, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        backend=backend,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        metric=metric,
        dask_client=dask_client,
    )

    # We cannot easily mock a function sent to dask
    # so for this test we create the whole set of models/ensembles
    # but prevent it to be loaded
    automl.fit(
        X_train, Y_train, task=MULTICLASS_CLASSIFICATION,
    )
    automl._backend.load_ensemble = unittest.mock.MagicMock(return_value=None)

    # A memory error occurs in the ensemble construction
    assert automl._backend.load_ensemble(automl._seed) is None

    # The load model is robust to this and loads the best model
    automl._load_models()
    assert automl.ensemble_ is not None

    # Just 1 model is there for ensemble and all weight must be on it
    get_models_with_weights = automl.get_models_with_weights()
    assert len(get_models_with_weights) == 1
    assert get_models_with_weights[0][0] == 1.0

    # Match a toy dataset
    if metric.name == 'balanced_accuracy':
        assert automl.score(X_test, Y_test) > 0.9
    elif metric.name == 'log_loss':
        assert automl.score(X_test, Y_test) <= 0.2
    else:
        raise ValueError(metric.name)

    del automl


def test_fail_if_feat_type_on_pandas_input(backend, dask_client):
    """We do not support feat type when pandas
    is provided as an input
    """
    automl = autosklearn.automl.AutoML(
        backend=backend,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        metric=accuracy,
        dask_client=dask_client,
    )

    X_train = pd.DataFrame({'a': [1, 1], 'c': [1, 2]})
    y_train = [1, 0]
    with pytest.raises(
        ValueError,
        match=""
        "providing the option feat_type to the fit method is not supported when using a Dataframe"
    ):
        automl.fit(
            X_train, y_train,
            task=BINARY_CLASSIFICATION,
            feat_type=['Categorical', 'Numerical'],
        )


def test_fail_if_dtype_changes_automl(backend, dask_client):
    """We do not support changes in the input type.
    Once a estimator is fitted, it should not change data type
    """
    automl = autosklearn.automl.AutoML(
        backend=backend,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        metric=accuracy,
        dask_client=dask_client,
    )

    X_train = pd.DataFrame({'a': [1, 1], 'c': [1, 2]})
    y_train = [1, 0]
    automl.InputValidator.validate(X_train, y_train, is_classification=True)
    with pytest.raises(
        ValueError,
        match="Auto-sklearn previously received features of type"
    ):
        automl.fit(
            X_train.to_numpy(), y_train,
            task=BINARY_CLASSIFICATION,
        )


@pytest.mark.parametrize(
    'memory_limit,task',
    [
        (memory_limit, task)
        for task in itertools.chain(CLASSIFICATION_TASKS, REGRESSION_TASKS)
        for memory_limit in (1, 10)
    ]
)
def test_subsample_if_too_large(memory_limit, task):
    fixture = {
        BINARY_CLASSIFICATION: {1: 436, 10: 569},
        MULTICLASS_CLASSIFICATION: {1: 204, 10: 1797},
        MULTILABEL_CLASSIFICATION: {1: 204, 10: 1797},
        REGRESSION: {1: 1310, 10: 1326},
        MULTIOUTPUT_REGRESSION: {1: 1310, 10: 1326}
    }
    mock = unittest.mock.Mock()
    if task == BINARY_CLASSIFICATION:
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    elif task == MULTICLASS_CLASSIFICATION:
        X, y = sklearn.datasets.load_digits(return_X_y=True)
    elif task == MULTILABEL_CLASSIFICATION:
        X, y_ = sklearn.datasets.load_digits(return_X_y=True)
        y = np.zeros((X.shape[0], 10))
        for i, j in enumerate(y_):
            y[i, j] = 1
    elif task == REGRESSION:
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        X = np.vstack((X, X, X))
        y = np.vstack((y.reshape((-1, 1)), y.reshape((-1, 1)), y.reshape((-1, 1))))
    elif task == MULTIOUTPUT_REGRESSION:
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        y = np.vstack((y, y)).transpose()
        X = np.vstack((X, X, X))
        y = np.vstack((y, y, y))
    else:
        raise ValueError(task)

    assert X.shape[0] == y.shape[0]

    X_new, y_new = AutoML.subsample_if_too_large(X, y, mock, 1, memory_limit, task)
    assert X_new.shape[0] == fixture[task][memory_limit]
    if memory_limit == 1:
        assert mock.warning.call_count == 1
    else:
        assert mock.warning.call_count == 0
