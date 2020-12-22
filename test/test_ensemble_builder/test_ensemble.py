import os
import sys
import time
import unittest.mock
import pickle
import pytest
import shutil

import dask.distributed
import numpy as np
import pandas as pd
from smac.runhistory.runhistory import RunValue, RunKey, RunHistory

from autosklearn.constants import MULTILABEL_CLASSIFICATION, BINARY_CLASSIFICATION
from autosklearn.metrics import roc_auc, accuracy, log_loss
from autosklearn.ensemble_builder import (
    EnsembleBuilder,
    EnsembleBuilderManager,
    Y_ENSEMBLE,
    Y_VALID,
    Y_TEST,
)
from autosklearn.ensembles.singlebest_ensemble import SingleBest

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from ensemble_utils import BackendMock, compare_read_preds, EnsembleBuilderMemMock, MockMetric  # noqa (E402: module level import not   at top of file)


@pytest.fixture(scope="function")
def ensemble_backend(request):
    test_id = '%s_%s' % (request.module.__name__, request.node.name)
    test_dir = os.path.join(this_directory, test_id)

    try:
        shutil.rmtree(test_dir)
    except:  # noqa E722
        pass

    # Make sure the folders we wanna create do not already exist.
    backend = BackendMock(test_dir)

    def get_finalizer(ensemble_backend):
        def session_run_at_end():
            try:
                shutil.rmtree(test_dir)
            except:  # noqa E722
                pass
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
    assert len(ensbuilder.read_preds) == 3, ensbuilder.read_preds.keys()
    assert len(ensbuilder.read_scores) == 3, ensbuilder.read_scores.keys()

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_1_0.0/predictions_ensemble_0_1_0.0.npy"
    )
    assert ensbuilder.read_scores[filename]["ens_score"] == 0.5

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy"
    )
    assert ensbuilder.read_scores[filename]["ens_score"] == 1.0


@pytest.mark.parametrize(
    "ensemble_nbest,max_models_on_disc,exp",
    (
            (1, None, 1),
            (1.0, None, 2),
            (0.1, None, 1),
            (0.9, None, 1),
            (1, 2, 1),
            (2, 1, 1),
    )
)
def testNBest(ensemble_backend, ensemble_nbest, max_models_on_disc, exp):
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
    )

    ensbuilder.score_ensemble_preds()
    sel_keys = ensbuilder.get_n_best_preds()

    assert len(sel_keys) == exp

    fixture = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy"
    )
    assert sel_keys[0] == fixture


@pytest.mark.parametrize("test_case,exp", [
    # If None, no reduction
    (None, 2),
    # If Int, limit only on exceed
    (4, 2),
    (1, 1),
    # If Float, translate float to # models.
    # below, mock of each file is 100 Mb and 4 files .model and .npy (test/val/pred) exist
    # per run (except for run3, there they are 5). Now, it takes 500MB for run 3 and
    # another 500 MB of slack because we keep as much space as the largest model
    # available as slack
    (1499.0, 1),
    (1500.0, 2),
    (9999.0, 2),
])
def testMaxModelsOnDisc(ensemble_backend, test_case, exp):
    ensemble_nbest = 4
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
        assert len(sel_keys) == exp, test_case


def testMaxModelsOnDisc2(ensemble_backend):
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
        ensbuilder.read_scores['pred'+str(i)] = {
            'ens_score': i*10,
            'num_run': i,
            'loaded': 1,
            "seed": 1,
            "disc_space_cost_mb": 50*i,
        }
        ensbuilder.read_preds['pred'+str(i)] = {Y_ENSEMBLE: True}
    sel_keys = ensbuilder.get_n_best_preds()
    assert ['pred49', 'pred48', 'pred47'] == sel_keys

    # Make sure at least one model is kept alive
    ensbuilder.max_models_on_disc = 0.0
    sel_keys = ensbuilder.get_n_best_preds()
    assert ['pred49'] == sel_keys


@pytest.mark.parametrize(
    "performance_range_threshold,exp",
    ((0.0, 4), (0.1, 4), (0.3, 3), (0.5, 2), (0.6, 2), (0.8, 1), (1.0, 1), (1, 1))
)
def testPerformanceRangeThreshold(ensemble_backend, performance_range_threshold, exp):
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=100,
        performance_range_threshold=performance_range_threshold
    )
    ensbuilder.read_scores = {
        'A': {'ens_score': 1, 'num_run': 1, 'loaded': -1, "seed": 1},
        'B': {'ens_score': 2, 'num_run': 2, 'loaded': -1, "seed": 1},
        'C': {'ens_score': 3, 'num_run': 3, 'loaded': -1, "seed": 1},
        'D': {'ens_score': 4, 'num_run': 4, 'loaded': -1, "seed": 1},
        'E': {'ens_score': 5, 'num_run': 5, 'loaded': -1, "seed": 1},
    }
    ensbuilder.read_preds = {
        key: {key_2: True for key_2 in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for key in ensbuilder.read_scores
    }
    sel_keys = ensbuilder.get_n_best_preds()

    assert len(sel_keys) == exp


@pytest.mark.parametrize(
    "performance_range_threshold,ensemble_nbest,exp",
    (
        (0.0, 1, 1), (0.0, 1.0, 4), (0.1, 2, 2), (0.3, 4, 3),
        (0.5, 1, 1), (0.6, 10, 2), (0.8, 0.5, 1), (1, 1.0, 1)
    )
)
def testPerformanceRangeThresholdMaxBest(ensemble_backend, performance_range_threshold,
                                         ensemble_nbest, exp):
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
    ensbuilder.read_scores = {
        'A': {'ens_score': 1, 'num_run': 1, 'loaded': -1, "seed": 1},
        'B': {'ens_score': 2, 'num_run': 2, 'loaded': -1, "seed": 1},
        'C': {'ens_score': 3, 'num_run': 3, 'loaded': -1, "seed": 1},
        'D': {'ens_score': 4, 'num_run': 4, 'loaded': -1, "seed": 1},
        'E': {'ens_score': 5, 'num_run': 5, 'loaded': -1, "seed": 1},
    }
    ensbuilder.read_preds = {
        key: {key_2: True for key_2 in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for key in ensbuilder.read_scores
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
    print()
    print(ensbuilder.read_preds.keys())
    print(ensbuilder.read_scores.keys())
    print(ensemble_backend.temporary_directory)

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy"
    )
    ensbuilder.read_scores[filename]["ens_score"] = -1

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_3_100.0/predictions_ensemble_0_3_100.0.npy"
    )
    ensbuilder.read_scores[filename]["ens_score"] = -1

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_1_0.0/predictions_ensemble_0_1_0.0.npy"
    )
    ensbuilder.read_scores[filename]["ens_score"] = -1

    sel_keys = ensbuilder.get_n_best_preds()

    fixture = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_1_0.0/predictions_ensemble_0_1_0.0.npy"
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
        ".auto-sklearn/runs/0_1_0.0/predictions_ensemble_0_1_0.0.npy"
    )
    d2 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy"
    )
    d3 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_3_100.0/predictions_ensemble_0_3_100.0.npy"
    )

    sel_keys = ensbuilder.get_n_best_preds()
    assert len(sel_keys) == 1
    ensbuilder.get_valid_test_preds(selected_keys=sel_keys)

    # Number of read files should be three and
    # predictions_ensemble_0_4_0.0.npy must not be in there
    assert len(ensbuilder.read_preds) == 3
    assert os.path.join(
            ensemble_backend.temporary_directory,
            ".auto-sklearn/runs/0_4_0.0/predictions_ensemble_0_4_0.0.npy"
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
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy"
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


def test_main(ensemble_backend):

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

    run_history, ensemble_nbest, _, _, _ = ensbuilder.main(
        time_left=np.inf, iteration=1, return_predictions=False,
    )

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

    assert os.path.exists(
        os.path.join(ensemble_backend.internals_directory, 'ensemble_read_preds.pkl')
    ), os.listdir(ensemble_backend.internals_directory)
    assert os.path.exists(
        os.path.join(ensemble_backend.internals_directory, 'ensemble_read_scores.pkl')
    ), os.listdir(ensemble_backend.internals_directory)


def test_run_end_at(ensemble_backend):
    with unittest.mock.patch('pynisher.enforce_limits') as pynisher_mock:
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

        current_time = time.time()

        ensbuilder.run(end_at=current_time + 10, iteration=1)
        # 4 seconds left because: 10 seconds - 5 seconds overhead - very little overhead,
        # but then rounded to an integer
        assert pynisher_mock.call_args_list[0][1]["wall_time_in_s"], 4


def testLimit(ensemble_backend):
    ensbuilder = EnsembleBuilderMemMock(backend=ensemble_backend,
                                        dataset_name="TEST",
                                        task_type=BINARY_CLASSIFICATION,
                                        metric=roc_auc,
                                        seed=0,  # important to find the test files
                                        ensemble_nbest=10,
                                        # small to trigger MemoryException
                                        memory_limit=100,
                                        )
    ensbuilder.SAVE2DISC = False

    read_scores_file = os.path.join(
        ensemble_backend.internals_directory,
        'ensemble_read_scores.pkl'
    )
    read_preds_file = os.path.join(
        ensemble_backend.internals_directory,
        'ensemble_read_preds.pkl'
    )

    def mtime_mock(filename):
        mtimes = {
            'predictions_ensemble_0_1_0.0.npy': 0,
            'predictions_valid_0_1_0.0.npy': 0.1,
            'predictions_test_0_1_0.0.npy': 0.2,
            'predictions_ensemble_0_2_0.0.npy': 1,
            'predictions_valid_0_2_0.0.npy': 1.1,
            'predictions_test_0_2_0.0.npy': 1.2,
            'predictions_ensemble_0_3_100.0.npy': 2,
            'predictions_valid_0_3_100.0.npy': 2.1,
            'predictions_test_0_3_100.0.npy': 2.2,
        }
        return mtimes[os.path.split(filename)[1]]

    with unittest.mock.patch('logging.getLogger') as get_logger_mock, \
            unittest.mock.patch('logging.config.dictConfig') as _, \
            unittest.mock.patch('os.path.getmtime') as mtime:
        logger_mock = unittest.mock.Mock()
        logger_mock.handlers = []
        get_logger_mock.return_value = logger_mock
        mtime.side_effect = mtime_mock

        ensbuilder.run(time_left=1000, iteration=0, pynisher_context='fork')
        assert os.path.exists(read_scores_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 1
        ensbuilder.run(time_left=1000, iteration=0, pynisher_context='fork')
        assert os.path.exists(read_scores_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 2
        ensbuilder.run(time_left=1000, iteration=0, pynisher_context='fork')
        assert os.path.exists(read_scores_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 3

        # it should try to reduce ensemble_nbest until it also failed at 2
        assert ensbuilder.ensemble_nbest == 1

        ensbuilder.run(time_left=1000, iteration=0, pynisher_context='fork')
        assert os.path.exists(read_scores_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 4

        # it should next reduce the number of models to read at most
        assert ensbuilder.read_at_most == 1

        # And then it still runs, but basically won't do anything any more except for raising error
        # messages via the logger
        ensbuilder.run(time_left=1000, iteration=0)
        assert os.path.exists(read_scores_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 4

        # In the previous assert, reduction is tried until failure
        # So that means we should have more than 1 memoryerror message
        assert logger_mock.error.call_count >= 1, "{}".format(
            logger_mock.error.call_args_list
        )
        for i in range(len(logger_mock.error.call_args_list)):
            assert 'Memory Exception -- Unable to further reduce' in str(
                logger_mock.error.call_args_list[i])


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

    ensbuilder.main(time_left=np.inf, iteration=1, return_predictions=False)

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

    ensemble_memory_file = os.path.join(
        ensemble_backend.internals_directory,
        'ensemble_read_scores.pkl'
    )
    assert os.path.exists(ensemble_memory_file)

    # Make sure we pickle the correct read scores
    with (open(ensemble_memory_file, "rb")) as memory:
        read_scores = pickle.load(memory)

    compare_read_preds(read_scores, ensbuilder.read_scores)

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
    compare_read_preds(ensbuilder2.read_scores, ensbuilder.read_scores)
    assert ensbuilder2.last_hash == ensbuilder.last_hash


@pytest.mark.parametrize("metric", [log_loss, accuracy])
@unittest.mock.patch('os.path.exists')
def test_get_identifiers_from_run_history(exists, metric, ensemble_run_history, ensemble_backend):
    exists.return_value = True
    ensemble = SingleBest(
         metric=log_loss,
         seed=1,
         run_history=ensemble_run_history,
         backend=ensemble_backend,
    )

    # Just one model
    assert len(ensemble.identifiers_) == 1

    # That model must be the best
    seed, num_run, budget = ensemble.identifiers_[0]
    assert num_run == 3
    assert seed == 1
    assert budget == 3.0


def test_ensemble_builder_process_realrun(dask_client_single_worker, ensemble_backend):
    manager = EnsembleBuilderManager(
        start_time=time.time(),
        time_left_for_ensembles=1000,
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
    )
    manager.build_ensemble(dask_client_single_worker)
    future = manager.futures.pop()
    dask.distributed.wait([future])  # wait for the ensemble process to finish
    result = future.result()
    history, _, _, _, _ = result

    assert 'ensemble_optimization_score' in history[0]
    assert history[0]['ensemble_optimization_score'] == 0.9
    assert 'ensemble_val_score' in history[0]
    assert history[0]['ensemble_val_score'] == 0.9
    assert 'ensemble_test_score' in history[0]
    assert history[0]['ensemble_test_score'] == 0.9


def test_ensemble_builder_nbest_remembered(
    ensemble_backend,
    dask_client_single_worker,
):
    """
    Makes sure ensemble builder returns the size of the ensemble that pynisher allowed
    This way, we can remember it and not waste more time trying big ensemble sizes
    """

    manager = EnsembleBuilderManager(
        start_time=time.time(),
        time_left_for_ensembles=1000,
        backend=ensemble_backend,
        dataset_name='Test',
        task=MULTILABEL_CLASSIFICATION,
        metric=roc_auc,
        ensemble_size=50,
        ensemble_nbest=10,
        max_models_on_disc=None,
        seed=0,
        precision=32,
        read_at_most=np.inf,
        ensemble_memory_limit=1000,
        random_state=0,
        max_iterations=None,
    )

    manager.build_ensemble(dask_client_single_worker, unit_test=True)
    future = manager.futures[0]
    dask.distributed.wait([future])  # wait for the ensemble process to finish
    assert future.result() == ([], 5, None, None, None)
    file_path = os.path.join(ensemble_backend.internals_directory, 'ensemble_read_preds.pkl')
    assert not os.path.exists(file_path)

    manager.build_ensemble(dask_client_single_worker, unit_test=True)

    future = manager.futures[0]
    dask.distributed.wait([future])  # wait for the ensemble process to finish
    assert not os.path.exists(file_path)
    assert future.result() == ([], 2, None, None, None)
