# -*- encoding: utf-8 -*-
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
from autosklearn.util.logging_ import setup_logger, get_logger
from autosklearn.constants import MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION, REGRESSION
from smac.tae import StatusType

sys.path.append(os.path.dirname(__file__))
from base import Base, extract_msg_from_log, count_succeses  # noqa (E402: module level import not at top of file)


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


def test_fit_roar(dask_client, backend):
    def get_roar_object_callback(
            scenario_dict,
            seed,
            ta,
            ta_kwargs,
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
        )

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    automl = autosklearn.automl.AutoML(
        backend=backend,
        time_left_for_this_task=30,
        per_run_time_limit=5,
        initial_configurations_via_metalearning=0,
        get_smac_object_callback=get_roar_object_callback,
        metric=accuracy,
        dask_client=dask_client,
    )
    setup_logger()
    automl._logger = get_logger('test_fit_roar')
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
        time_left_for_this_task=45,
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
        # default to 10 to make sure we delete models.
        max_models_on_disc=8,
    )

    automl.fit(X, Y, task=MULTICLASS_CLASSIFICATION,
               X_test=X, y_test=Y)

    # Assert at least one model file has been deleted and that there were no
    # deletion errors
    log_file_path = glob.glob(os.path.join(
        backend.temporary_directory, 'AutoML(' + str(seed) + '):*.log'))
    with open(log_file_path[0]) as log_file:
        log_content = log_file.read()
        assert 'Deleted files of non-candidate model' in log_content
        assert 'Failed to delete files of non-candidate model' not in log_content
        assert 'Failed to lock model' not in log_content

    # Assert that the files of the models used by the ensemble weren't deleted
    model_files = backend.list_all_models(seed=seed)
    model_files_idx = set()
    for m_file in model_files:
        # Extract the model identifiers from the filename
        m_file = os.path.split(m_file)[1].replace('.model', '').split('.', 2)
        model_files_idx.add((int(m_file[0]), int(m_file[1]), float(m_file[2])))
    ensemble_members_idx = set(automl.ensemble_.identifiers_)
    assert ensemble_members_idx.issubset(model_files_idx)

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
    setup_logger()
    auto._logger = get_logger('test_automl_outputs')
    auto.fit(
        X=X_train,
        y=Y_train,
        X_test=X_test,
        y_test=Y_test,
        dataset_name=name,
        task=MULTICLASS_CLASSIFICATION,
    )

    # Log file path
    log_file_path = glob.glob(os.path.join(
        backend.temporary_directory, 'AutoML*.log'))[0]

    # pickled data manager (without one hot encoding!)
    with open(data_manager_file, 'rb') as fh:
        D = pickle.load(fh)
        assert np.allclose(D.data['X_train'], X_train)

    # Check that all directories are there
    fixture = ['cv_models', 'true_targets_ensemble.npy',
               'start_time_100', 'datamanager.pkl',
               'predictions_ensemble', 'ensemble_read_preds.pkl',
               'done', 'ensembles', 'predictions_test', 'models']
    assert (
        sorted(os.listdir(os.path.join(backend.temporary_directory, '.auto-sklearn')))
        == sorted(fixture)
    )

    # At least one ensemble, one validation, one test prediction and one
    # model and one ensemble
    fixture = os.listdir(os.path.join(backend.temporary_directory,
                                      '.auto-sklearn', 'predictions_ensemble'))
    assert len(fixture) > 0

    fixture = glob.glob(os.path.join(backend.temporary_directory, '.auto-sklearn',
                                     'models', '100.*.model'))
    assert len(fixture) > 0

    fixture = os.listdir(os.path.join(backend.temporary_directory,
                                      '.auto-sklearn', 'ensembles'))
    assert '100.0000000001.ensemble' in fixture

    # Start time
    start_time_file_path = os.path.join(backend.temporary_directory,
                                        '.auto-sklearn', "start_time_100")
    with open(start_time_file_path, 'r') as fh:
        start_time = float(fh.read())
    assert time.time() - start_time >= 10, extract_msg_from_log(log_file_path)

    del auto


def test_do_dummy_prediction(backend, dask_client):
    datasets = {
        'breast_cancer': BINARY_CLASSIFICATION,
        'wine': MULTICLASS_CLASSIFICATION,
        'diabetes': REGRESSION,
    }

    for name, task in datasets.items():

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
        setup_logger()
        auto._logger = get_logger('test_do_dummy_predictions')
        auto._backend.save_datamanager(datamanager)
        D = backend.load_datamanager()

        # Check if data manager is correcly loaded
        assert D.info['task'] == datamanager.info['task']

        auto._do_dummy_prediction(D, 1)

        # Ensure that the dummy predictions are not in the current working
        # directory, but in the temporary directory.
        assert not os.path.exists(os.path.join(os.getcwd(), '.auto-sklearn'))
        assert os.path.exists(os.path.join(
            backend.temporary_directory, '.auto-sklearn', 'predictions_ensemble',
            'predictions_ensemble_1_1_0.0.npy')
        )

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
    setup_logger()
    auto._logger = get_logger('test_fail_if_dummy_prediction_fails')
    auto._backend._make_internals_directory()
    auto._backend.save_datamanager(datamanager)

    # First of all, check that ta.run() is actually called.
    ta_run_mock.return_value = StatusType.SUCCESS, None, None, "test"
    auto._do_dummy_prediction(datamanager, 1)
    ta_run_mock.assert_called_once_with(1, cutoff=time_for_this_task)

    # Case 1. Check that function raises no error when statustype == success.
    # ta.run() returns status, cost, runtime, and additional info.
    ta_run_mock.return_value = StatusType.SUCCESS, None, None, "test"
    raised = False
    try:
        auto._do_dummy_prediction(datamanager, 1)
    except ValueError:
        raised = True
    assert not raised, 'Exception raised'

    # Case 2. Check that if statustype returned by ta.run() != success,
    # the function raises error.
    ta_run_mock.return_value = StatusType.CRASHED, None, None, "test"
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.CRASHED and additional output: test.'
    ):
        auto._do_dummy_prediction( datamanager, 1)

    ta_run_mock.return_value = StatusType.ABORT, None, None, "test"
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.ABORT '
              'and additional output: test.',
    ):
        auto._do_dummy_prediction(datamanager, 1)

    ta_run_mock.return_value = StatusType.TIMEOUT, None, None, "test"
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.TIMEOUT '
              'and additional output: test.'
    ):
        auto._do_dummy_prediction(datamanager, 1)

    ta_run_mock.return_value = StatusType.MEMOUT, None, None, "test"
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.MEMOUT '
              'and additional output: test.',
    ):
        auto._do_dummy_prediction(datamanager, 1)

    ta_run_mock.return_value = StatusType.CAPPED, None, None, "test"
    with pytest.raises(
        ValueError,
        match='Dummy prediction failed with run state StatusType.CAPPED '
              'and additional output: test.'
    ):
        auto._do_dummy_prediction(datamanager, 1)


@unittest.mock.patch('autosklearn.smbo.AutoMLSMBO.run_smbo')
def test_exceptions_inside_log_in_smbo(smbo_run_mock, backend, dask_client):

    automl = autosklearn.automl.AutoML(
        backend,
        20,
        5,
        metric=accuracy,
        dask_client=dask_client,
    )

    output_file = 'test_exceptions_inside_log.log'
    setup_logger(output_file=output_file)
    logger = get_logger('test_exceptions_inside_log')

    # Create a custom exception to prevent other errors to slip in
    class MyException(Exception):
        pass

    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    # The first call is on dummy predictor failure
    message = str(np.random.randint(100)) + '_run_smbo'
    smbo_run_mock.side_effect = MyException(message)

    with unittest.mock.patch('autosklearn.automl.AutoML._get_logger') as mock:
        mock.return_value = logger
        with pytest.raises(MyException):
            automl.fit(
                X_train,
                Y_train,
                task=MULTICLASS_CLASSIFICATION,
            )
        with open(output_file) as f:
            assert message in f.read()

    # Cleanup
    os.unlink(output_file)


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
        match="feat_type cannot be provided when using pandas"
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
