# -*- encoding: utf-8 -*-
import itertools
import os
import pickle
import sys
import time
import glob
import unittest
import unittest.mock
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import sklearn.datasets
from sklearn.ensemble import VotingRegressor, VotingClassifier
from smac.scenario.scenario import Scenario
from smac.facade.roar_facade import ROAR

from autosklearn.automl import AutoML, AutoMLClassifier, AutoMLRegressor, _model_predict
from autosklearn.data.validation import InputValidator
import autosklearn.automl
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.metrics import (
    accuracy, log_loss, balanced_accuracy, default_metric_for_task
)
from autosklearn.evaluation.abstract_evaluator import MyDummyClassifier, MyDummyRegressor
from autosklearn.util.logging_ import PickableLoggerAdapter
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
from automl_utils import print_debug_information, count_succeses, AutoMLLogParser, includes_all_scores, includes_train_scores, performance_over_time_is_plausible  # noqa (E402: module level import not at top of file)


class AutoMLStub(AutoML):
    def __init__(self):
        self.__class__ = AutoML
        self._task = None
        self._dask_client = None
        self._is_dask_client_internally_created = False

    def __del__(self):
        pass


def test_fit(dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        seed=0,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        metric=accuracy,
        dask_client=dask_client,
    )

    automl.fit(X_train, Y_train, task=MULTICLASS_CLASSIFICATION)

    score = automl.score(X_test, Y_test)
    assert score > 0.8
    assert count_succeses(automl.cv_results_) > 0
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert performance_over_time_is_plausible(automl.performance_over_time_) is True
    assert automl._task == MULTICLASS_CLASSIFICATION

    del automl


def test_fit_roar(dask_client_single_worker):
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
        time_left_for_this_task=30,
        per_run_time_limit=5,
        initial_configurations_via_metalearning=0,
        get_smac_object_callback=get_roar_object_callback,
        metric=accuracy,
        dask_client=dask_client_single_worker,
    )

    automl.fit(X_train, Y_train, task=MULTICLASS_CLASSIFICATION)

    score = automl.score(X_test, Y_test)
    assert score > 0.8
    assert count_succeses(automl.cv_results_) > 0
    assert includes_train_scores(automl.performance_over_time_.columns) is True
    assert automl._task == MULTICLASS_CLASSIFICATION

    del automl


def test_refit_shuffle_on_fail(dask_client):

    failing_model = unittest.mock.Mock()
    failing_model.fit.side_effect = [ValueError(), ValueError(), None]
    failing_model.fit_transformer.side_effect = [
        ValueError(), ValueError(), (None, {})]
    failing_model.get_max_iter.return_value = 100

    auto = AutoML(30, 5, dask_client=dask_client)
    ensemble_mock = unittest.mock.Mock()
    ensemble_mock.get_selected_model_identifiers.return_value = [(1, 1, 50.0)]
    auto.ensemble_ = ensemble_mock
    auto.InputValidator = InputValidator()
    for budget_type in [None, 'iterations']:
        auto._budget_type = budget_type

        auto.models_ = {(1, 1, 50.0): failing_model}

        # Make sure a valid 2D array is given to automl
        X = np.array([1, 2, 3]).reshape(-1, 1)
        y = np.array([1, 2, 3])
        auto.InputValidator.fit(X, y)
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


def test_delete_non_candidate_models(dask_client):

    seed = 555
    X, Y, _, _ = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        delete_tmp_folder_after_terminate=False,
        time_left_for_this_task=60,
        per_run_time_limit=5,
        ensemble_nbest=3,
        seed=seed,
        initial_configurations_via_metalearning=0,
        resampling_strategy='holdout',
        include={
            'classifier': ['sgd'],
            'feature_preprocessor': ['no_preprocessing']
        },
        metric=accuracy,
        dask_client=dask_client,
        # Force model to be deleted. That is, from 50 which is the
        # default to 3 to make sure we delete models.
        max_models_on_disc=3,
    )

    automl.fit(X, Y, task=MULTICLASS_CLASSIFICATION, X_test=X, y_test=Y)

    # Assert at least one model file has been deleted and that there were no
    # deletion errors
    log_file_path = glob.glob(os.path.join(
        automl._backend.temporary_directory, 'AutoML(' + str(seed) + '):*.log'))
    with open(log_file_path[0]) as log_file:
        log_content = log_file.read()
        assert 'Deleted files of non-candidate model' in log_content, log_content
        assert 'Failed to delete files of non-candidate model' not in log_content, log_content
        assert 'Failed to lock model' not in log_content, log_content

    # Assert that the files of the models used by the ensemble weren't deleted
    model_files = automl._backend.list_all_models(seed=seed)
    model_files_idx = set()
    for m_file in model_files:
        # Extract the model identifiers from the filename
        m_file = os.path.split(m_file)[1].replace('.model', '').split('.', 2)
        model_files_idx.add((int(m_file[0]), int(m_file[1]), float(m_file[2])))
    ensemble_members_idx = set(automl.ensemble_.identifiers_)
    assert ensemble_members_idx.issubset(model_files_idx), (ensemble_members_idx, model_files_idx)

    del automl


def test_binary_score_and_include(dask_client):
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
        20, 5,
        include={'classifier': ['sgd'],
                 'feature_preprocessor': ['no_preprocessing']},
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


def test_automl_outputs(dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    name = 'iris'
    auto = autosklearn.automl.AutoML(
        30, 5,
        initial_configurations_via_metalearning=0,
        seed=100,
        metric=accuracy,
        dask_client=dask_client,
        delete_tmp_folder_after_terminate=False,
    )

    auto.fit(
        X=X_train,
        y=Y_train,
        X_test=X_test,
        y_test=Y_test,
        dataset_name=name,
        task=MULTICLASS_CLASSIFICATION,
    )

    data_manager_file = os.path.join(
        auto._backend.temporary_directory,
        '.auto-sklearn',
        'datamanager.pkl'
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
        'ensemble_read_losses.pkl',
        'runs',
        'ensembles',
        'ensemble_history.json',
    ]
    assert (
        sorted(os.listdir(os.path.join(auto._backend.temporary_directory,
                                       '.auto-sklearn')))
        == sorted(fixture)
    )

    # At least one ensemble, one validation, one test prediction and one
    # model and one ensemble
    fixture = glob.glob(os.path.join(
        auto._backend.temporary_directory,
        '.auto-sklearn', 'runs', '*', 'predictions_ensemble*npy',
    ))
    assert len(fixture) > 0

    fixture = glob.glob(os.path.join(auto._backend.temporary_directory, '.auto-sklearn',
                                     'runs', '*', '100.*.model'))
    assert len(fixture) > 0

    fixture = os.listdir(os.path.join(auto._backend.temporary_directory,
                                      '.auto-sklearn', 'ensembles'))
    assert '100.0000000000.ensemble' in fixture

    # Start time
    start_time_file_path = os.path.join(auto._backend.temporary_directory,
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
def test_do_dummy_prediction(dask_client, datasets):

    name, task = datasets

    X_train, Y_train, X_test, Y_test = putil.get_dataset(name)
    datamanager = XYDataManager(
        X_train, Y_train,
        X_test, Y_test,
        task=task,
        dataset_name=name,
        feat_type={i: 'numerical' for i in range(X_train.shape[1])},
    )

    auto = autosklearn.automl.AutoML(
        20, 5,
        initial_configurations_via_metalearning=25,
        metric=accuracy,
        dask_client=dask_client,
        delete_tmp_folder_after_terminate=False,
    )
    auto._backend = auto._create_backend()

    # Make a dummy logger
    auto._logger_port = 9020
    auto._logger = unittest.mock.Mock()
    auto._logger.info.return_value = None

    auto._backend.save_datamanager(datamanager)
    D = auto._backend.load_datamanager()

    # Check if data manager is correcly loaded
    assert D.info['task'] == datamanager.info['task']
    auto._do_dummy_prediction(D, 1)

    # Ensure that the dummy predictions are not in the current working
    # directory, but in the temporary directory.
    unexpected_directory = os.path.join(os.getcwd(), '.auto-sklearn')
    expected_directory = os.path.join(
        auto._backend.temporary_directory,
        '.auto-sklearn',
        'runs',
        '1_1_0.0',
        'predictions_ensemble_1_1_0.0.npy'
    )
    assert not os.path.exists(unexpected_directory)
    assert os.path.exists(expected_directory)

    auto._clean_logger()

    del auto


@unittest.mock.patch('autosklearn.evaluation.ExecuteTaFuncWithQueue.run')
def test_fail_if_dummy_prediction_fails(ta_run_mock, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    datamanager = XYDataManager(
        X_train, Y_train,
        X_test, Y_test,
        task=2,
        feat_type={i: 'Numerical' for i in range(X_train.shape[1])},
        dataset_name='iris',
    )

    time_for_this_task = 30
    per_run_time = 10
    auto = autosklearn.automl.AutoML(time_for_this_task,
                                     per_run_time,
                                     initial_configurations_via_metalearning=25,
                                     metric=accuracy,
                                     dask_client=dask_client,
                                     delete_tmp_folder_after_terminate=False,
                                     )
    auto._backend = auto._create_backend()
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
def test_exceptions_inside_log_in_smbo(smbo_run_mock, dask_client):

    # Below importing and shutdown is a workaround, to make sure
    # we reset the port to collect messages. Randomly, when running
    # this test with multiple other test at the same time causes this
    # test to fail. This resets the singletons of the logging class
    import logging
    logging.shutdown()

    automl = autosklearn.automl.AutoML(
        20,
        5,
        metric=accuracy,
        dask_client=dask_client,
        delete_tmp_folder_after_terminate=False,
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
    logger_name = 'AutoML(%d):%s' % (1, dataset_name)
    logger = logging.getLogger(logger_name)
    logfile = os.path.join(automl._backend.temporary_directory, logger_name + '.log')
    assert os.path.exists(logfile), print_debug_information(automl) + str(automl._clean_logger())

    # Give some time for the error message to be printed in the
    # log file
    found_message = False
    for incr_tolerance in range(5):
        with open(logfile) as f:
            lines = f.readlines()
        if any(message in line for line in lines):
            found_message = True
            break
        else:
            time.sleep(incr_tolerance)

    # Speed up the closing after forced crash
    automl._clean_logger()

    if not found_message:
        pytest.fail("Did not find {} in the log file {} for logger {}/{}/{}".format(
            message,
            print_debug_information(automl),
            vars(automl._logger.logger),
            vars(logger),
            vars(logging.getLogger())
        ))


@pytest.mark.parametrize("metric", [log_loss, balanced_accuracy])
def test_load_best_individual_model(metric, dask_client):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        metric=metric,
        dask_client=dask_client,
        delete_tmp_folder_after_terminate=False,
    )

    # We cannot easily mock a function sent to dask
    # so for this test we create the whole set of models/ensembles
    # but prevent it to be loaded
    automl.fit(X_train, Y_train, task=MULTICLASS_CLASSIFICATION)

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
        # Seen values in github actions of 0.6978304740364537
        assert automl.score(X_test, Y_test) < 0.7
    else:
        raise ValueError(metric.name)

    del automl


def test_fail_if_feat_type_on_pandas_input(dask_client):
    """We do not support feat type when pandas
    is provided as an input
    """
    automl = autosklearn.automl.AutoML(
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
            feat_type={1: 'Categorical', 2: 'Numerical'},
        )


@pytest.mark.parametrize(
    'memory_limit,precision,task',
    [
        (memory_limit, precision, task)
        for task in itertools.chain(CLASSIFICATION_TASKS, REGRESSION_TASKS)
        for precision in (float, np.float32, np.float64, np.float128)
        for memory_limit in (1, 100, None)
    ]
)
def test_subsample_if_too_large(memory_limit, precision, task):
    fixture = {
        BINARY_CLASSIFICATION: {
            1: {float: 1310, np.float32: 2621, np.float64: 1310, np.float128: 655},
            100: {float: 12000, np.float32: 12000, np.float64: 12000, np.float128: 12000},
            None: {float: 12000, np.float32: 12000, np.float64: 12000, np.float128: 12000},
        },
        MULTICLASS_CLASSIFICATION: {
            1: {float: 204, np.float32: 409, np.float64: 204, np.float128: 102},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
            None: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        MULTILABEL_CLASSIFICATION: {
            1: {float: 204, np.float32: 409, np.float64: 204, np.float128: 102},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
            None: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        REGRESSION: {
            1: {float: 655, np.float32: 1310, np.float64: 655, np.float128: 327},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
            None: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        },
        MULTIOUTPUT_REGRESSION: {
            1: {float: 655, np.float32: 1310, np.float64: 655, np.float128: 327},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
            None: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        }
    }
    mock = unittest.mock.Mock()
    if task == BINARY_CLASSIFICATION:
        X, y = sklearn.datasets.make_hastie_10_2()
    elif task == MULTICLASS_CLASSIFICATION:
        X, y = sklearn.datasets.load_digits(return_X_y=True)
    elif task == MULTILABEL_CLASSIFICATION:
        X, y_ = sklearn.datasets.load_digits(return_X_y=True)
        y = np.zeros((X.shape[0], 10))
        for i, j in enumerate(y_):
            y[i, j] = 1
    elif task == REGRESSION:
        X, y = sklearn.datasets.make_friedman1(n_samples=5000, n_features=20)
    elif task == MULTIOUTPUT_REGRESSION:
        X, y = sklearn.datasets.make_friedman1(n_samples=5000, n_features=20)
        y = np.vstack((y, y)).transpose()
    else:
        raise ValueError(task)
    X = X.astype(precision)

    assert X.shape[0] == y.shape[0]

    X_new, y_new = AutoML.subsample_if_too_large(X, y, mock, 1, memory_limit, task)
    assert X_new.shape[0] == fixture[task][memory_limit][precision]
    if memory_limit == 1:
        if precision in (np.float128, np.float64, float):
            assert mock.warning.call_count == 2
        else:
            assert mock.warning.call_count == 1
    else:
        assert mock.warning.call_count == 0


def data_input_and_target_types():
    n_rows = 100

    # Create valid inputs
    X_ndarray = np.random.random(size=(n_rows, 5))
    X_ndarray[X_ndarray < .9] = 0

    # Binary Classificaiton
    y_binary_ndarray = np.random.random(size=n_rows)
    y_binary_ndarray[y_binary_ndarray >= 0.5] = 1
    y_binary_ndarray[y_binary_ndarray < 0.5] = 0

    # Multiclass classification
    y_multiclass_ndarray = np.random.random(size=n_rows)
    y_multiclass_ndarray[y_multiclass_ndarray > 0.66] = 2
    y_multiclass_ndarray[(y_multiclass_ndarray <= 0.66) & (y_multiclass_ndarray >= 0.33)] = 1
    y_multiclass_ndarray[y_multiclass_ndarray < 0.33] = 0

    # Multilabel classificaiton
    y_multilabel_ndarray = np.random.random(size=(n_rows, 3))
    y_multilabel_ndarray[y_multilabel_ndarray > 0.5] = 1
    y_multilabel_ndarray[y_multilabel_ndarray <= 0.5] = 0

    # Regression
    y_regression_ndarray = np.random.random(size=n_rows)

    # Multioutput Regression
    y_multioutput_regression_ndarray = np.random.random(size=(n_rows, 3))

    xs = [
        X_ndarray,
        list(X_ndarray),
        csr_matrix(X_ndarray),
        pd.DataFrame(data=X_ndarray),
    ]

    ys_binary = [
        y_binary_ndarray,
        list(y_binary_ndarray),
        csr_matrix(y_binary_ndarray),
        pd.Series(y_binary_ndarray),
        pd.DataFrame(data=y_binary_ndarray),
    ]

    ys_multiclass = [
        y_multiclass_ndarray,
        list(y_multiclass_ndarray),
        csr_matrix(y_multiclass_ndarray),
        pd.Series(y_multiclass_ndarray),
        pd.DataFrame(data=y_multiclass_ndarray),
    ]

    ys_multilabel = [
        y_multilabel_ndarray,
        list(y_multilabel_ndarray),
        csr_matrix(y_multilabel_ndarray),
        # pd.Series(y_multilabel_ndarray)
        pd.DataFrame(data=y_multilabel_ndarray),
    ]

    ys_regression = [
        y_regression_ndarray,
        list(y_regression_ndarray),
        csr_matrix(y_regression_ndarray),
        pd.Series(y_regression_ndarray),
        pd.DataFrame(data=y_regression_ndarray),
    ]

    ys_multioutput_regression = [
        y_multioutput_regression_ndarray,
        list(y_multioutput_regression_ndarray),
        csr_matrix(y_multioutput_regression_ndarray),
        # pd.Series(y_multioutput_regression_ndarray),
        pd.DataFrame(data=y_multioutput_regression_ndarray),
    ]

    # [ (X, y, X_test, y_test, task), ... ]
    return (
        (X, y, X, y, task)
        for X in xs
        for y, task in itertools.chain(
            itertools.product(ys_binary, [BINARY_CLASSIFICATION]),
            itertools.product(ys_multiclass, [MULTICLASS_CLASSIFICATION]),
            itertools.product(ys_multilabel, [MULTILABEL_CLASSIFICATION]),
            itertools.product(ys_regression, [REGRESSION]),
            itertools.product(ys_multioutput_regression, [MULTIOUTPUT_REGRESSION]),
        )
    )


@pytest.mark.parametrize("X, y, X_test, y_test, task", data_input_and_target_types())
def test_input_and_target_types(dask_client, X, y, X_test, y_test, task):

    if task in CLASSIFICATION_TASKS:
        automl = AutoMLClassifier(
            time_left_for_this_task=15,
            per_run_time_limit=5,
            dask_client=dask_client,
        )
    else:
        automl = AutoMLRegressor(
            time_left_for_this_task=15,
            per_run_time_limit=5,
            dask_client=dask_client,
        )
    # To save time fitting and only validate the inputs we only return
    # the configuration space
    automl.fit(
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        only_return_configuration_space=True
    )
    assert automl._task == task
    assert automl._metric.name == default_metric_for_task[task].name


def data_test_model_predict_outsputs_correct_shapes():
    datasets = sklearn.datasets
    binary = datasets.make_classification(
        n_samples=5, n_classes=2, random_state=0
    )
    multiclass = datasets.make_classification(
        n_samples=5, n_informative=3, n_classes=3, random_state=0
    )
    multilabel = datasets.make_multilabel_classification(
        n_samples=5, n_classes=3, random_state=0
    )
    regression = datasets.make_regression(
        n_samples=5, random_state=0
    )
    multioutput = datasets.make_regression(
        n_samples=5, n_targets=3, random_state=0
    )

    # TODO issue 1169
    #   While testing output shapes, realised all models are wrapped to provide
    #   a special predict_proba that outputs a different shape than usual.
    #   This includes DummyClassifier and DummyRegressor which are wrapped as
    #   `MyDummyClassifier/Regressor` and require a config object.
    #   config == 1 : Classifier uses 'uniform', Regressor uses 'mean'
    #   else        : Classifier uses 'most_frequent', Regressor uses 'median'
    #
    #   This wrapping of probabilities with
    #       `convert_multioutput_multiclass_to_multilabel`
    #   can probably be just put into a base class which queries subclasses
    #   as to whether it's needed.
    #
    #   tldr; thats why we use MyDummyX here instead of the default models
    #         from sklearn
    def classifier(X, y):
        return MyDummyClassifier(config=1, random_state=0).fit(X, y)

    def regressor(X, y):
        return MyDummyRegressor(config=1, random_state=0).fit(X, y)

    # How cross validation models are currently grouped together
    def voting_classifier(X, y):
        classifiers = [
            MyDummyClassifier(config=1, random_state=0).fit(X, y)
            for _ in range(5)
        ]
        vc = VotingClassifier(estimators=None, voting='soft')
        vc.estimators_ = classifiers
        return vc

    def voting_regressor(X, y):
        regressors = [
            MyDummyRegressor(config=1, random_state=0).fit(X, y)
            for _ in range(5)
        ]
        vr = VotingRegressor(estimators=None)
        vr.estimators_ = regressors
        return vr

    test_data = {
        BINARY_CLASSIFICATION: {
            'models': [classifier(*binary), voting_classifier(*binary)],
            'data': binary,
            # prob of false/true for the one class
            'expected_output_shape': (len(binary[0]), 2)
        },
        MULTICLASS_CLASSIFICATION: {
            'models': [classifier(*multiclass), voting_classifier(*multiclass)],
            'data': multiclass,
            # prob of true for each possible class
            'expected_output_shape': (len(multiclass[0]), 3)
        },
        MULTILABEL_CLASSIFICATION: {
            'models': [classifier(*multilabel), voting_classifier(*multilabel)],
            'data': multilabel,
            # probability of true for each binary label
            'expected_output_shape': (len(multilabel[0]), 3)  # type: ignore
        },
        REGRESSION: {
            'models': [regressor(*regression), voting_regressor(*regression)],
            'data': regression,
            # array of single outputs
            'expected_output_shape': (len(regression[0]), )
        },
        MULTIOUTPUT_REGRESSION: {
            'models': [regressor(*multioutput), voting_regressor(*multioutput)],
            'data': multioutput,
            # array of vector otuputs
            'expected_output_shape': (len(multioutput[0]), 3)
        }
    }

    return itertools.chain.from_iterable(
        [
            (model, cfg['data'], task, cfg['expected_output_shape'])
            for model in cfg['models']
        ]
        for task, cfg in test_data.items()
    )


@pytest.mark.parametrize(
    "model, data, task, expected_output_shape",
    data_test_model_predict_outsputs_correct_shapes()
)
def test_model_predict_outputs_correct_shapes(model, data, task, expected_output_shape):
    X, y = data
    prediction = _model_predict(model=model, X=X, task=task)
    assert prediction.shape == expected_output_shape


def test_model_predict_outputs_warnings_to_logs():
    X = list(range(20))
    task = REGRESSION
    logger = PickableLoggerAdapter('test_model_predict_correctly_outputs_warnings')
    logger.warning = unittest.mock.Mock()

    class DummyModel:
        def predict(self, x):
            warnings.warn('test warning', Warning)
            return x

    model = DummyModel()

    _model_predict(model, X, task, logger=logger)

    assert logger.warning.call_count == 1, "Logger should have had warning called"


def test_model_predict_outputs_to_stdout_if_no_logger():
    X = list(range(20))
    task = REGRESSION

    class DummyModel:
        def predict(self, x):
            warnings.warn('test warning', Warning)
            return x

    model = DummyModel()

    with warnings.catch_warnings(record=True) as w:
        _model_predict(model, X, task, logger=None)

        assert len(w) == 1, "One warning sould have been emmited"


@pytest.mark.parametrize("task, y", [
    (BINARY_CLASSIFICATION, np.asarray(
        9999 * [0] + 1 * [1]
    )),
    (MULTICLASS_CLASSIFICATION, np.asarray(
        4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4]
    )),
    (MULTILABEL_CLASSIFICATION, np.asarray(
        4999 * [[0, 1, 1]] + 4999 * [[1, 1, 0]] + 1 * [[1, 0, 1]] + 1 * [[0, 0, 0]]
    ))
])
def test_subsample_classification_unique_labels_stay_in_training_set(task, y):
    n_samples = 10000
    X = np.random.random(size=(n_samples, 3))
    memory_limit = 1  # Force subsampling
    mock = unittest.mock.Mock()

    # Make sure our test assumptions are correct
    assert len(y) == n_samples, "Ensure tests are correctly setup"

    values, counts = np.unique(y, axis=0, return_counts=True)
    unique_labels = [value for value, count in zip(values, counts) if count == 1]
    assert len(unique_labels), "Ensure we have unique labels in the test"

    _, y_sampled = AutoML.subsample_if_too_large(X, y,
                                                 logger=mock,
                                                 seed=1,
                                                 memory_limit=memory_limit,
                                                 task=task)

    assert len(y_sampled) <= len(y), \
        "Ensure sampling took place"
    assert all(label in y_sampled for label in unique_labels), \
        "All unique labels present in the return sampled set"
