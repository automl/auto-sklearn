import glob
import os
import sys
import time
import threading
import unittest.mock
import pickle
import pytest

import numpy as np

import pandas as pd

from smac.runhistory.runhistory import RunValue, RunKey, RunHistory

from autosklearn.constants import MULTILABEL_CLASSIFICATION, BINARY_CLASSIFICATION
from autosklearn.metrics import roc_auc, accuracy, log_loss
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.ensemble_builder import (
    EnsembleBuilder,
    ensemble_builder_process,
    fit_and_return_ensemble,
    Y_VALID,
    Y_TEST,
)
from autosklearn.ensembles.singlebest_ensemble import SingleBest

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from ensemble_utils import BackendMock, compare_read_preds, EnsembleBuilderMemMock, MockMetric  # noqa (E402: module level import not   at top of file)


@pytest.fixture(scope="function")
def ensemble_backend(request):

    # Make sure the folders we wanna create do not already exist.
    backend = BackendMock()
    ensemble_memory_file = os.path.join(
        backend.internals_directory,
        'ensemble_read_preds.pkl'
    )

    # We make sure that every test starts without
    # any memory file
    if os.path.exists(ensemble_memory_file):
        os.unlink(ensemble_memory_file)

    def get_finalizer(ensemble_backend):
        def session_run_at_end():

            # Remove the memory file created by the ensemble builder
            ensemble_memory_file = os.path.join(
                ensemble_backend.internals_directory,
                'ensemble_read_preds.pkl'
            )
            if os.path.exists(ensemble_memory_file):
                os.unlink(ensemble_memory_file)

            # Remove the log file if created in the test/test_ensemble/data
            # area
            logfile = glob.glob(os.path.join(
                ensemble_backend.temporary_directory, '*.log'))
            if len(logfile) > 1 and os.path.exists(logfile[0]):
                os.unlink(logfile[0])
        return session_run_at_end
    request.addfinalizer(get_finalizer(backend))

    return backend


@pytest.fixture(scope="function")
def ensemble_run_history(request):

    run_history = RunHistory()
    run_history._add(
        RunKey(
            config_id=3,
            instance_id='{"task_id": "breast_cancer"}',
            seed=1,
            budget=3.0
        ),
        RunValue(
            cost=0.11347517730496459,
            time=0.21858787536621094,
            status=None,
            starttime=time.time(),
            endtime=time.time(),
            additional_info={
                'duration': 0.20323538780212402,
                'num_run': 3,
                'configuration_origin': 'Random Search'}
        ),
        status=None,
        origin=None,
    )
    run_history._add(
        RunKey(
            config_id=6,
            instance_id='{"task_id": "breast_cancer"}',
            seed=1,
            budget=6.0
        ),
        RunValue(
            cost=2*0.11347517730496459,
            time=2*0.21858787536621094,
            status=None,
            starttime=time.time(),
            endtime=time.time(),
            additional_info={
                'duration': 0.20323538780212402,
                'num_run': 6,
                'configuration_origin': 'Random Search'}
        ),
        status=None,
        origin=None,
    )
    return run_history


def testRead(ensemble_backend):

    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
    )

    success = ensbuilder.score_ensemble_preds()
    assert success, str(ensbuilder.read_preds)
    assert len(ensbuilder.read_preds) == 3

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
    )
    assert ensbuilder.read_preds[filename]["ens_score"] == 0.5

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
    )
    assert ensbuilder.read_preds[filename]["ens_score"] == 1.0


def testNBest(ensemble_backend):
    for ensemble_nbest, models_on_disc, exp in (
            (1, None, 1),
            (1.0, None, 2),
            (0.1, None, 1),
            (0.9, None, 1),
            (1, 2, 1),
            (2, 1, 1),
    ):
        ensbuilder = EnsembleBuilder(
            backend=ensemble_backend,
            dataset_name="TEST",
            task_type=BINARY_CLASSIFICATION,
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=ensemble_nbest,
            max_models_on_disc=models_on_disc,
        )

        ensbuilder.score_ensemble_preds()
        sel_keys = ensbuilder.get_n_best_preds()

        assert len(sel_keys) == exp

        fixture = os.path.join(
            ensemble_backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
        )
        assert sel_keys[0] == fixture


def testMaxModelsOnDisc(ensemble_backend):

    ensemble_nbest = 4
    for (test_case, exp) in [
            # If None, no reduction
            (None, 2),
            # If Int, limit only on exceed
            (4, 2),
            (1, 1),
            # If Float, translate float to # models.
            # below, mock of each file is 100 Mb and
            # 4 files .model and .npy (test/val/pred) exist
            (700.0, 1),
            (800.0, 2),
            (9999.0, 2),
    ]:
        ensbuilder = EnsembleBuilder(
            backend=ensemble_backend,
            dataset_name="TEST",
            task_type=BINARY_CLASSIFICATION,
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=ensemble_nbest,
            max_models_on_disc=test_case,
        )

        with unittest.mock.patch('os.path.getsize') as mock:
            mock.return_value = 100*1024*1024
            ensbuilder.score_ensemble_preds()
            sel_keys = ensbuilder.get_n_best_preds()
            assert len(sel_keys) == exp

    # Test for Extreme scenarios
    # Make sure that the best predictions are kept
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=50,
        max_models_on_disc=10000.0,
    )
    ensbuilder.read_preds = {}
    for i in range(50):
        ensbuilder.read_preds['pred'+str(i)] = {
            'ens_score': i*10,
            'num_run': i,
            0: True,
            'loaded': 1,
            "seed": 1,
            "disc_space_cost_mb": 50*i,
        }
    sel_keys = ensbuilder.get_n_best_preds()
    assert ['pred49', 'pred48', 'pred47', 'pred46'] == sel_keys

    # Make sure at least one model is kept alive
    ensbuilder.max_models_on_disc = 0.0
    sel_keys = ensbuilder.get_n_best_preds()
    assert ['pred49'] == sel_keys


def testPerformanceRangeThreshold(ensemble_backend):
    to_test = ((0.0, 4), (0.1, 4), (0.3, 3), (0.5, 2), (0.6, 2), (0.8, 1),
               (1.0, 1), (1, 1))
    for performance_range_threshold, exp in to_test:
        ensbuilder = EnsembleBuilder(
            backend=ensemble_backend,
            dataset_name="TEST",
            task_type=BINARY_CLASSIFICATION,
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=100,
            performance_range_threshold=performance_range_threshold
        )
        ensbuilder.read_preds = {
            'A': {'ens_score': 1, 'num_run': 1, 0: True, 'loaded': -1, "seed": 1},
            'B': {'ens_score': 2, 'num_run': 2, 0: True, 'loaded': -1, "seed": 1},
            'C': {'ens_score': 3, 'num_run': 3, 0: True, 'loaded': -1, "seed": 1},
            'D': {'ens_score': 4, 'num_run': 4, 0: True, 'loaded': -1, "seed": 1},
            'E': {'ens_score': 5, 'num_run': 5, 0: True, 'loaded': -1, "seed": 1},
        }
        sel_keys = ensbuilder.get_n_best_preds()

        assert len(sel_keys) == exp


def testPerformanceRangeThresholdMaxBest(ensemble_backend):
    to_test = ((0.0, 1, 1), (0.0, 1.0, 4), (0.1, 2, 2), (0.3, 4, 3),
               (0.5, 1, 1), (0.6, 10, 2), (0.8, 0.5, 1), (1, 1.0, 1))
    for performance_range_threshold, ensemble_nbest, exp in to_test:
        ensbuilder = EnsembleBuilder(
            backend=ensemble_backend,
            dataset_name="TEST",
            task_type=BINARY_CLASSIFICATION,
            metric=roc_auc,
            seed=0,  # important to find the test files
            ensemble_nbest=ensemble_nbest,
            performance_range_threshold=performance_range_threshold,
            max_models_on_disc=None,
        )
        ensbuilder.read_preds = {
            'A': {'ens_score': 1, 'num_run': 1, 0: True, 'loaded': -1, "seed": 1},
            'B': {'ens_score': 2, 'num_run': 2, 0: True, 'loaded': -1, "seed": 1},
            'C': {'ens_score': 3, 'num_run': 3, 0: True, 'loaded': -1, "seed": 1},
            'D': {'ens_score': 4, 'num_run': 4, 0: True, 'loaded': -1, "seed": 1},
            'E': {'ens_score': 5, 'num_run': 5, 0: True, 'loaded': -1, "seed": 1},
        }
        sel_keys = ensbuilder.get_n_best_preds()

        assert len(sel_keys) == exp


def testFallBackNBest(ensemble_backend):

    ensbuilder = EnsembleBuilder(backend=ensemble_backend,
                                 dataset_name="TEST",
                                 task_type=BINARY_CLASSIFICATION,
                                 metric=roc_auc,
                                 seed=0,  # important to find the test files
                                 ensemble_nbest=1
                                 )

    ensbuilder.score_ensemble_preds()

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
    )
    ensbuilder.read_preds[filename]["ens_score"] = -1

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_3_100.0.npy"
    )
    ensbuilder.read_preds[filename]["ens_score"] = -1

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
    )
    ensbuilder.read_preds[filename]["ens_score"] = -1

    sel_keys = ensbuilder.get_n_best_preds()

    fixture = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
    )
    assert len(sel_keys) == 1
    assert sel_keys[0] == fixture


def testGetValidTestPreds(ensemble_backend):

    ensbuilder = EnsembleBuilder(backend=ensemble_backend,
                                 dataset_name="TEST",
                                 task_type=BINARY_CLASSIFICATION,
                                 metric=roc_auc,
                                 seed=0,  # important to find the test files
                                 ensemble_nbest=1
                                 )

    ensbuilder.score_ensemble_preds()

    d1 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_1_0.0.npy"
    )
    d2 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
    )
    d3 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_3_100.0.npy"
    )

    sel_keys = ensbuilder.get_n_best_preds()
    assert len(sel_keys) == 1
    ensbuilder.get_valid_test_preds(selected_keys=sel_keys)

    # Number of read files should be three and
    # predictions_ensemble_0_4_0.0.npy must not be in there
    assert len(ensbuilder.read_preds) == 3
    assert os.path.join(
            ensemble_backend.temporary_directory,
            ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_4_0.0.npy"
    ) not in ensbuilder.read_preds

    # not selected --> should still be None
    assert ensbuilder.read_preds[d1][Y_VALID] is None
    assert ensbuilder.read_preds[d1][Y_TEST] is None
    assert ensbuilder.read_preds[d3][Y_VALID] is None
    assert ensbuilder.read_preds[d3][Y_TEST] is None

    # selected --> read valid and test predictions
    assert ensbuilder.read_preds[d2][Y_VALID] is not None
    assert ensbuilder.read_preds[d2][Y_TEST] is not None


def testEntireEnsembleBuilder(ensemble_backend):

    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=2,
    )
    ensbuilder.SAVE2DISC = False

    ensbuilder.score_ensemble_preds()

    d2 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/predictions_ensemble/predictions_ensemble_0_2_0.0.npy"
    )

    sel_keys = ensbuilder.get_n_best_preds()
    assert len(sel_keys) > 0

    ensemble = ensbuilder.fit_ensemble(selected_keys=sel_keys)
    print(ensemble, sel_keys)

    n_sel_valid, n_sel_test = ensbuilder.get_valid_test_preds(selected_keys=sel_keys)

    # both valid and test prediction files are available
    assert len(n_sel_valid) > 0
    assert n_sel_valid == n_sel_test

    y_valid = ensbuilder.predict(
        set_="valid",
        ensemble=ensemble,
        selected_keys=n_sel_valid,
        n_preds=len(sel_keys),
        index_run=1,
    )
    y_test = ensbuilder.predict(
        set_="test",
        ensemble=ensemble,
        selected_keys=n_sel_test,
        n_preds=len(sel_keys),
        index_run=1,
    )

    # predictions for valid and test are the same
    # --> should results in the same predictions
    np.testing.assert_array_almost_equal(y_valid, y_test)

    # since d2 provides perfect predictions
    # it should get a higher weight
    # so that y_valid should be exactly y_valid_d2
    y_valid_d2 = ensbuilder.read_preds[d2][Y_VALID][:, 1]
    np.testing.assert_array_almost_equal(y_valid, y_valid_d2)


def testMain(ensemble_backend):

    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=MULTILABEL_CLASSIFICATION,  # Multilabel Classification
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=2,
        max_models_on_disc=None,
        )
    ensbuilder.SAVE2DISC = False

    run_history, ensemble_nbest, sleep = ensbuilder.main(time_left=np.inf, iteration=1)

    # It is the first time we create the ensemble, so no sleep
    assert not sleep

    assert len(ensbuilder.read_preds) == 3
    assert ensbuilder.last_hash is not None
    assert ensbuilder.y_true_ensemble is not None

    # Make sure the run history is ok

    # We expect at least 1 element to be in the ensemble
    assert len(run_history) > 0

    # As the data loader loads the same val/train/test
    # we expect 1.0 as score and all keys available
    expected_performance = {
        'ensemble_val_score': 1.0,
        'ensemble_test_score': 1.0,
        'ensemble_optimization_score': 1.0,
    }

    # Make sure that expected performance is a subset of the run history
    assert all(item in run_history[0].items() for item in expected_performance.items())
    assert 'Timestamp' in run_history[0]
    assert isinstance(run_history[0]['Timestamp'], pd.Timestamp)

    # If we try to create the ensemble on the same data, no new
    # ensemble will be created so we should sleep
    run_history, ensemble_nbest, sleep = ensbuilder.main(time_left=np.inf, iteration=2)
    assert sleep


def testLimit(ensemble_backend):
    ensbuilder = EnsembleBuilderMemMock(backend=ensemble_backend,
                                        dataset_name="TEST",
                                        task_type=BINARY_CLASSIFICATION,
                                        metric=roc_auc,
                                        seed=0,  # important to find the test files
                                        ensemble_nbest=10,
                                        # small to trigger MemoryException
                                        memory_limit=10
                                        )
    ensbuilder.SAVE2DISC = False

    ensbuilder.run(time_left=1000, iteration=0)

    # it should try to reduce ensemble_nbest until it also failed at 2
    assert ensbuilder.ensemble_nbest == 1


def test_read_pickle_read_preds(ensemble_backend):
    """
    This procedure test that we save the read predictions before
    destroying the ensemble builder and that we are able to read
    them safely after
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=MULTILABEL_CLASSIFICATION,  # Multilabel Classification
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=2,
        max_models_on_disc=None,
        )
    ensbuilder.SAVE2DISC = False

    run_history, ensemble_nbest, sleep = ensbuilder.main(time_left=np.inf, iteration=1)

    # Check that the memory was created
    ensemble_memory_file = os.path.join(
        ensemble_backend.internals_directory,
        'ensemble_read_preds.pkl'
    )
    assert os.path.exists(ensemble_memory_file)

    # Make sure we pickle the correct read preads and hash
    with (open(ensemble_memory_file, "rb")) as memory:
        read_preds, last_hash = pickle.load(memory)

    compare_read_preds(read_preds, ensbuilder.read_preds)
    assert last_hash == ensbuilder.last_hash

    # Then create a new instance, which should automatically read this file
    ensbuilder2 = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=MULTILABEL_CLASSIFICATION,  # Multilabel Classification
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=2,
        max_models_on_disc=None,
        )
    compare_read_preds(ensbuilder2.read_preds, ensbuilder.read_preds)
    assert ensbuilder2.last_hash == ensbuilder.last_hash


def testPredict():
    # Test that ensemble prediction applies weights correctly to given
    # predictions. There are two possible cases:
    # 1) predictions.shape[0] == len(self.weights_). In this case,
    # predictions include those made by zero-weighted models. Therefore,
    # we simply apply each weights to the corresponding model preds.
    # 2) predictions.shape[0] < len(self.weights_). In this case,
    # predictions exclude those made by zero-weighted models. Therefore,
    # we first exclude all occurrences of zero in self.weights_, and then
    # apply the weights.
    # If none of the above is the case, predict() raises Error.
    ensemble = EnsembleSelection(ensemble_size=3,
                                 task_type=BINARY_CLASSIFICATION,
                                 random_state=np.random.RandomState(0),
                                 metric=accuracy,
                                 )
    # Test for case 1. Create (3, 2, 2) predictions.
    per_model_pred = np.array([
        [[0.9, 0.1],
         [0.4, 0.6]],
        [[0.8, 0.2],
         [0.3, 0.7]],
        [[1.0, 0.0],
         [0.1, 0.9]]
    ])
    # Weights of 3 hypothetical models
    ensemble.weights_ = [0.7, 0.2, 0.1]
    pred = ensemble.predict(per_model_pred)
    truth = np.array([[0.89, 0.11],  # This should be the true prediction.
                      [0.35, 0.65]])
    assert np.allclose(pred, truth)

    # Test for case 2.
    per_model_pred = np.array([
        [[0.9, 0.1],
         [0.4, 0.6]],
        [[0.8, 0.2],
         [0.3, 0.7]],
        [[1.0, 0.0],
         [0.1, 0.9]]
    ])
    # The third model now has weight of zero.
    ensemble.weights_ = [0.7, 0.2, 0.0, 0.1]
    pred = ensemble.predict(per_model_pred)
    truth = np.array([[0.89, 0.11],
                      [0.35, 0.65]])
    assert np.allclose(pred, truth)

    # Test for error case.
    per_model_pred = np.array([
        [[0.9, 0.1],
         [0.4, 0.6]],
        [[0.8, 0.2],
         [0.3, 0.7]],
        [[1.0, 0.0],
         [0.1, 0.9]]
    ])
    # Now the weights have 2 zero weights and 2 non-zero weights,
    # which is incompatible.
    ensemble.weights_ = [0.6, 0.0, 0.0, 0.4]

    with pytest.raises(ValueError):
        ensemble.predict(per_model_pred)


@pytest.mark.parametrize("metric", [log_loss, accuracy])
@unittest.mock.patch('os.path.exists')
def test_get_identifiers_from_run_history(exists, metric, ensemble_run_history):
    exists.return_value = True
    ensemble = SingleBest(
         metric=log_loss,
         random_state=1,
         run_history=ensemble_run_history,
         model_dir='/tmp',
    )

    # Just one model
    assert len(ensemble.identifiers_) == 1

    # That model must be the best
    seed, num_run, budget = ensemble.identifiers_[0]
    assert num_run == 3
    assert seed == 1
    assert budget == 3.0


def test_ensemble_builder_process_termination_request(dask_client, ensemble_backend):
    """
    Makes sure we can kill an ensemble process via a event
    """

    # We need a dask_client for the event
    ensemble_termination_request = threading.Event()

    # Set the event so the run does not even start
    ensemble_termination_request.set()

    ensemble = ensemble_builder_process(
        start_time=time.time(),
        time_left_for_ensembles=1000,
        sleep_duration=2,
        address=dask_client.scheduler_info()['address'],
        history=list(),
        event=ensemble_termination_request,
        backend=ensemble_backend,
        dataset_name='Test',
        task=BINARY_CLASSIFICATION,
        metric=roc_auc,
        ensemble_size=50,
        ensemble_nbest=10,
        max_models_on_disc=None,
        seed=0,
        precision=32,
        max_iterations=1,
        read_at_most=np.inf,
        ensemble_memory_limit=10,
        random_state=0,
        logger_name='Ensemblebuilder',
    )

    # make sure message is in log file
    msg = 'Terminating ensemble building as SMAC process is done'
    logfile = glob.glob(os.path.join(
        ensemble_backend.temporary_directory, '*.log'))[0]
    with open(logfile) as f:
        assert msg in f.read()

    # Also makes sure the ensemble does not return any history
    assert ensemble == []


def test_ensemble_builder_process_realrun(dask_client, ensemble_backend):
    history = ensemble_builder_process(
        start_time=time.time(),
        time_left_for_ensembles=1000,
        sleep_duration=2,
        event=None,
        history=list(),
        address=dask_client.scheduler_info()['address'],
        backend=ensemble_backend,
        dataset_name='Test',
        task=BINARY_CLASSIFICATION,
        metric=MockMetric,
        ensemble_size=50,
        ensemble_nbest=10,
        max_models_on_disc=None,
        seed=0,
        precision=32,
        max_iterations=1,
        read_at_most=np.inf,
        ensemble_memory_limit=None,
        random_state=0,
        logger_name='Ensemblebuilder',
    )

    assert 'ensemble_optimization_score' in history[0]
    assert history[0]['ensemble_optimization_score'] == 0.9
    assert 'ensemble_val_score' in history[0]
    assert history[0]['ensemble_val_score'] == 0.9
    assert 'ensemble_test_score' in history[0]
    assert history[0]['ensemble_test_score'] == 0.9


@unittest.mock.patch('autosklearn.ensemble_builder.EnsembleBuilder.fit_ensemble')
def test_ensemble_builder_nbest_remembered(fit_ensemble, ensemble_backend):
    """
    Makes sure ensemble builder returns the size of the ensemble that pynisher allowed
    This way, we can remember it and not waste more time trying big ensemble sizes
    """

    def register_ensemble_sizes_per_call(selected_keys):
        var = [np.ones([2500, 2500]) for a in selected_keys]  # noqa: F841
    fit_ensemble.side_effect = register_ensemble_sizes_per_call

    fit_ensemble.return_value = None

    ensemble_history, ensemble_nbest, sleep = fit_and_return_ensemble(
        time_left=1000,
        backend=ensemble_backend,
        dataset_name='Test',
        task_type=MULTILABEL_CLASSIFICATION,
        metric=roc_auc,
        ensemble_size=50,
        ensemble_nbest=10,
        max_models_on_disc=None,
        seed=0,
        precision=32,
        iteration=0,
        read_at_most=np.inf,
        memory_limit=1000,
        random_state=0,
        logger_name='Ensemblebuilder',
    )

    # The ensemble n size must be one, because in 1Gb
    # it can only fit 1 out of the 2 models as each one
    # involves an overhead of an array of 2500x2500
    assert ensemble_nbest == 1
