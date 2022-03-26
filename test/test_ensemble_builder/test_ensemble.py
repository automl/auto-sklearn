from __future__ import annotations

import os
import pickle
import time

import dask.distributed
import numpy as np
import pandas as pd
from smac.runhistory.runhistory import RunHistory, RunKey, RunValue

from autosklearn.automl_common.common.ensemble_building.abstract_ensemble import (
    AbstractEnsemble,
)
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION
from autosklearn.ensemble_builder import (
    Y_ENSEMBLE,
    Y_TEST,
    Y_VALID,
    EnsembleBuilder,
    EnsembleBuilderManager,
)
from autosklearn.ensembles.singlebest_ensemble import SingleBest
from autosklearn.metrics import Scorer, accuracy, log_loss, make_scorer, roc_auc

from pytest_cases import fixture, parametrize, parametrize_with_cases
from unittest.mock import Mock, patch

import test.test_ensemble_builder.cases as cases
from test.conftest import DEFAULT_SEED


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_read(ensemble_backend: Backend) -> None:
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
    )

    success = ensbuilder.compute_loss_per_model()
    assert success, f"read_preds = {str(ensbuilder.read_preds)}"

    assert len(ensbuilder.read_preds) == 3, ensbuilder.read_preds.keys()
    assert len(ensbuilder.read_losses) == 3, ensbuilder.read_losses.keys()

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_1_0.0/predictions_ensemble_0_1_0.0.npy",
    )
    assert ensbuilder.read_losses[filename]["ens_loss"] == 0.5

    filename = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy",
    )
    assert ensbuilder.read_losses[filename]["ens_loss"] == 0.0


@parametrize(
    "ensemble_nbest, max_models_on_disc, expected",
    (
        (1, None, 1),
        (1.0, None, 2),
        (0.1, None, 1),
        (0.9, None, 1),
        (1, 2, 1),
        (2, 1, 1),
    ),
)
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_nbest(
    ensemble_backend: Backend,
    ensemble_nbest: int | float,
    max_models_on_disc: int | None,
    expected: int,
) -> None:
    """
    Parameters
    ----------
    ensemble_backend: Backend
        The backend to use. In this case, we specifically rely on the `setup_3_models`
        setup.

    ensemble_nbest: int | float
        The parameter to use for consider the n best, int being absolute and float being
        fraction.

    max_models_on_disc: int | None
        The maximum amount of models to keep on disk

    expected: int
        The number of keys expected to be selected

    Expects
    -------
    * get_n_best_preds should contain 2 keys
    * The first key should be model 0_2_0_0
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
    )

    ensbuilder.compute_loss_per_model()
    sel_keys = ensbuilder.get_n_best_preds()

    assert len(sel_keys) == expected

    fixture = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy",
    )
    assert sel_keys[0] == fixture


@parametrize(
    "max_models_on_disc, expected",
    [
        # If None, no reduction
        (None, 2),
        # If Int, limit only on exceed
        (4, 2),
        (1, 1),
        # If Float, translate float to # models.
        # below, mock of each file is 100 Mb and 4 files .model and .npy (test/val/pred)
        # per run (except for run3, there they are 5). Now, it takes 500MB for run 3 and
        # another 500 MB of slack because we keep as much space as the largest model
        # available as slack
        (1499.0, 1),
        (1500.0, 2),
        (9999.0, 2),
    ],
)
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_max_models_on_disc(
    ensemble_backend: Backend,
    max_models_on_disc: int | float,
    expected: int,
) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The backend to use, relies on setup_3_models

    max_models_on_disc : int | float
        The max_models_on_disc param to use

    expected : int
        The expected number of selected models

    Expects
    -------
    * The number of selected models should be as expected
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=4,
        max_models_on_disc=max_models_on_disc,
    )

    with patch("os.path.getsize") as mock:
        mock.return_value = 100 * 1024 * 1024
        ensbuilder.compute_loss_per_model()
        sel_keys = ensbuilder.get_n_best_preds()
        assert len(sel_keys) == expected


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_max_models_on_disc_2(ensemble_backend: Backend) -> None:
    # Test for Extreme scenarios
    # Make sure that the best predictions are kept
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=50,
        max_models_on_disc=10000.0,
    )
    ensbuilder.read_preds = {}

    for n in range(50):
        loss = 10 * -n
        ensbuilder.read_losses["pred" + str(n)] = {
            "ens_loss": loss,
            "num_run": n,
            "loaded": 1,
            "seed": 0,
            "disc_space_cost_mb": 50 * n,
        }
        ensbuilder.read_preds["pred" + str(n)] = {Y_ENSEMBLE: True}

    sel_keys = ensbuilder.get_n_best_preds()
    assert ["pred49", "pred48", "pred47"] == sel_keys


@parametrize("n_models", [50, 10, 2, 1])
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_max_models_on_disc_preserves_always_preserves_at_least_one_model(
    n_models: int,
    ensemble_backend: Backend,
) -> None:
    """
    Parameters
    ----------
    n_models : int

    ensemble_backend : Backend

    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=50,
        max_models_on_disc=0.0,
    )

    read_losses = {
        f"pred{n}": {
            "ens_loss": 10 * -n,
            "num_run": n + 1,
            "loaded": 1,
            "seed": 0,
            "disc_space_cost_mb": 50 * n,
        }
        for n in range(n_models)
    }
    best_model = min(read_losses, key=lambda m: read_losses[m]["ens_loss"])

    ensbuilder.read_losses = read_losses
    ensbuilder.read_preds = {f"pred{n}": {Y_ENSEMBLE: True} for n in range(n_models)}

    sel_keys = ensbuilder.get_n_best_preds()
    assert [best_model] == sel_keys


@parametrize(
    "performance_range_threshold, expected_selected",
    ((0.0, 4), (0.1, 4), (0.3, 3), (0.5, 2), (0.6, 2), (0.8, 1), (1.0, 1)),
)
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_performance_range_threshold(
    ensemble_backend: Backend,
    performance_range_threshold: float,
    expected_selected: int,
) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The backend to use

    performance_range_threshold : float
        THe performance range threshold to use

    expected_selected : int
        The number of selected models for there to be

    Expects
    -------
    * Expects the given amount of models to be selected given a performance range
    threshold.
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=100,
        performance_range_threshold=performance_range_threshold,
    )
    ensbuilder.read_losses = {
        "A": {"ens_loss": -1, "num_run": 1, "loaded": -1, "seed": 1},
        "B": {"ens_loss": -2, "num_run": 2, "loaded": -1, "seed": 1},
        "C": {"ens_loss": -3, "num_run": 3, "loaded": -1, "seed": 1},
        "D": {"ens_loss": -4, "num_run": 4, "loaded": -1, "seed": 1},
        "E": {"ens_loss": -5, "num_run": 5, "loaded": -1, "seed": 1},
    }
    ensbuilder.read_preds = {
        name: {preds_key: True for preds_key in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for name in ensbuilder.read_losses
    }

    sel_keys = ensbuilder.get_n_best_preds()
    assert len(sel_keys) == expected_selected


@parametrize(
    "performance_range_threshold, ensemble_nbest, expected_selected",
    (
        (0.0, 1, 1),
        (0.0, 1.0, 4),
        (0.1, 2, 2),
        (0.3, 4, 3),
        (0.5, 1, 1),
        (0.6, 10, 2),
        (0.8, 0.5, 1),
        (1, 1.0, 1),
    ),
)
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_performance_range_threshold_with_ensemble_nbest(
    ensemble_backend: Backend,
    performance_range_threshold: float,
    ensemble_nbest: int | float,
    expected_selected: int,
) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
    performance_range_threshold : float
    ensemble_nbest : int | float
    expected_selected : int
        The number of models expected to be selected

    Expects
    -------
    * Given the setup of params for test_performance_range_threshold and ensemble_nbest,
    the expected number of models should be selected.
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=ensemble_nbest,
        performance_range_threshold=performance_range_threshold,
        max_models_on_disc=None,
    )
    ensbuilder.read_losses = {
        "A": {"ens_loss": -1, "num_run": 1, "loaded": -1, "seed": 1},
        "B": {"ens_loss": -2, "num_run": 2, "loaded": -1, "seed": 1},
        "C": {"ens_loss": -3, "num_run": 3, "loaded": -1, "seed": 1},
        "D": {"ens_loss": -4, "num_run": 4, "loaded": -1, "seed": 1},
        "E": {"ens_loss": -5, "num_run": 5, "loaded": -1, "seed": 1},
    }
    ensbuilder.read_preds = {
        name: {pred_name: True for pred_name in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for name in ensbuilder.read_losses
    }
    sel_keys = ensbuilder.get_n_best_preds()

    assert len(sel_keys) == expected_selected


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_fall_back_nbest(ensemble_backend: Backend) -> None:
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=1,
    )

    ensbuilder.compute_loss_per_model()
    print()
    print(ensbuilder.read_preds.keys())
    print(ensbuilder.read_losses.keys())
    print(ensemble_backend.temporary_directory)

    for model in ["0_1_0.0", "0_2_0.0", "0_3_100.0"]:
        filename = os.path.join(
            ensemble_backend.temporary_directory,
            f".auto-sklearn/runs/{model}/predictions_ensemble_{model}.npy",
        )
        ensbuilder.read_losses[filename]["ens_loss"] = -1

    sel_keys = ensbuilder.get_n_best_preds()

    best_model = "0_1_0.0"
    fixture = os.path.join(
        ensemble_backend.temporary_directory,
        f".auto-sklearn/runs/{best_model}/predictions_ensemble_{best_model}.npy",
    )

    assert len(sel_keys) == 1
    assert sel_keys[0] == fixture


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_get_valid_test_preds(ensemble_backend: Backend) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The ensemble backend to use with the setup_3_models setup
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=1,
    )

    # There are 3 models in the setup
    # * Run 1 is the dummy run
    # * Run 2 and Run 3 share the same predictions
    # -> Run 2 is selected with ensemble_nbest = 1
    paths = [
        os.path.join(
            ensemble_backend.temporary_directory,
            f".auto-sklearn/runs/{model}/predictions_ensemble_{model}.npy",
        )
        for model in ["0_1_0.0", "0_2_0.0", "0_3_100.0"]
    ]

    ensbuilder.compute_loss_per_model()

    sel_keys = ensbuilder.get_n_best_preds()
    assert len(sel_keys) == 1

    ensbuilder.get_valid_test_preds(selected_keys=sel_keys)

    # Number of read files should be three and contain those of the models in the setup
    assert set(ensbuilder.read_preds.keys()) == set(paths)

    selected = sel_keys
    non_selected = set(paths) - set(sel_keys)

    # not selected --> should still be None
    for key in non_selected:
        assert ensbuilder.read_preds[key][Y_VALID] is None
        assert ensbuilder.read_preds[key][Y_TEST] is None

    # selected --> read valid and test predictions
    for key in selected:
        assert ensbuilder.read_preds[key][Y_VALID] is not None
        assert ensbuilder.read_preds[key][Y_TEST] is not None


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_ensemble_builder_predictions(ensemble_backend: Backend) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The ensemble backend to use with the setup_3_models setup

    Expects
    -------
    * The validation and test sets should both have equal predictions for them?
    * Since model 0_2_0.0 has predictions exactly equal to the targets, it should
      recieve full weight and that the predictions should be identical to that models
      predictions
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=2,
    )
    ensbuilder.SAVE2DISC = False

    ensbuilder.compute_loss_per_model()

    d2 = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy",
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


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_main(ensemble_backend: Backend) -> None:
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=2,
        max_models_on_disc=None,
    )
    ensbuilder.SAVE2DISC = False

    run_history, ensemble_nbest, _, _, _ = ensbuilder.main(
        time_left=np.inf,
        iteration=1,
        return_predictions=False,
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
        "ensemble_val_score": 1.0,
        "ensemble_test_score": 1.0,
        "ensemble_optimization_score": 1.0,
    }

    # Make sure that expected performance is a subset of the run history
    assert all(item in run_history[0].items() for item in expected_performance.items())
    assert "Timestamp" in run_history[0]
    assert isinstance(run_history[0]["Timestamp"], pd.Timestamp)

    assert os.path.exists(
        os.path.join(ensemble_backend.internals_directory, "ensemble_read_preds.pkl")
    ), os.listdir(ensemble_backend.internals_directory)
    assert os.path.exists(
        os.path.join(ensemble_backend.internals_directory, "ensemble_read_losses.pkl")
    ), os.listdir(ensemble_backend.internals_directory)


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_run_end_at(ensemble_backend: Backend) -> None:
    with patch("pynisher.enforce_limits") as pynisher_mock:
        ensbuilder = EnsembleBuilder(
            backend=ensemble_backend,
            dataset_name="TEST",
            task_type=BINARY_CLASSIFICATION,
            metric=roc_auc,
            seed=DEFAULT_SEED,  # important to find the test files
            ensemble_nbest=2,
            max_models_on_disc=None,
        )
        ensbuilder.SAVE2DISC = False

        current_time = time.time()

        ensbuilder.run(
            end_at=current_time + 10,
            iteration=1,
            pynisher_context="forkserver",
        )
        # 4 seconds left because: 10 seconds - 5 seconds overhead - little overhead
        # but then rounded to an integer
        assert pynisher_mock.call_args_list[0][1]["wall_time_in_s"] == 4


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_limit(ensemble_backend: Backend) -> None:
    class EnsembleBuilderMemMock(EnsembleBuilder):
        def fit_ensemble(self, selected_keys):
            return True

        def predict(
            self,
            set_: str,
            ensemble: AbstractEnsemble,
            selected_keys: list,
            n_preds: int,
            index_run: int,
        ):
            np.ones([10000000, 1000000])

    ensbuilder = EnsembleBuilderMemMock(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=10,
        memory_limit=10,  # small to trigger MemoryException
    )

    ensbuilder.SAVE2DISC = False

    read_losses_file = os.path.join(
        ensemble_backend.internals_directory, "ensemble_read_losses.pkl"
    )
    read_preds_file = os.path.join(
        ensemble_backend.internals_directory, "ensemble_read_preds.pkl"
    )

    def mtime_mock(filename: str) -> float:
        mtimes = {
            "predictions_ensemble_0_1_0.0.npy": 0.0,
            "predictions_valid_0_1_0.0.npy": 0.1,
            "predictions_test_0_1_0.0.npy": 0.2,
            "predictions_ensemble_0_2_0.0.npy": 1.0,
            "predictions_valid_0_2_0.0.npy": 1.1,
            "predictions_test_0_2_0.0.npy": 1.2,
            "predictions_ensemble_0_3_100.0.npy": 2.0,
            "predictions_valid_0_3_100.0.npy": 2.1,
            "predictions_test_0_3_100.0.npy": 2.2,
        }
        return mtimes[os.path.split(filename)[1]]

    with patch("logging.getLogger") as get_logger_mock, patch(
        "os.path.getmtime"
    ) as mtime, patch("logging.config.dictConfig"):

        logger_mock = Mock()
        logger_mock.handlers = []
        get_logger_mock.return_value = logger_mock
        mtime.side_effect = mtime_mock

        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")
        assert os.path.exists(read_losses_file)
        assert not os.path.exists(read_preds_file)
        print(logger_mock.warning.call_args_list)
        assert logger_mock.warning.call_count == 1

        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")
        assert os.path.exists(read_losses_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 2

        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")
        assert os.path.exists(read_losses_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 3

        # it should try to reduce ensemble_nbest until it also failed at 2
        assert ensbuilder.ensemble_nbest == 1

        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")
        assert os.path.exists(read_losses_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 4

        # it should next reduce the number of models to read at most
        assert ensbuilder.read_at_most == 1

        # And then it still runs, but basically won't do anything any more except for
        # raising error messages via the logger
        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")
        assert os.path.exists(read_losses_file)
        assert not os.path.exists(read_preds_file)
        assert logger_mock.warning.call_count == 4

        # In the previous assert, reduction is tried until failure
        # So that means we should have more than 1 memoryerror message
        assert logger_mock.error.call_count >= 1, "{}".format(
            logger_mock.error.call_args_list
        )
        for i in range(len(logger_mock.error.call_args_list)):
            assert "Memory Exception -- Unable to further reduce" in str(
                logger_mock.error.call_args_list[i]
            )


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_read_pickle_read_preds(ensemble_backend: Backend) -> None:
    """
    This procedure test that we save the read predictions before
    destroying the ensemble builder and that we are able to read
    them safely after
    """
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=2,
        max_models_on_disc=None,
    )
    ensbuilder.SAVE2DISC = False

    ensbuilder.main(time_left=np.inf, iteration=1, return_predictions=False)

    # Check that the memory was created
    ensemble_memory_file = os.path.join(
        ensemble_backend.internals_directory, "ensemble_read_preds.pkl"
    )
    assert os.path.exists(ensemble_memory_file)

    # Make sure we pickle the correct read preads and hash
    with (open(ensemble_memory_file, "rb")) as memory:
        read_preds, last_hash = pickle.load(memory)

    def assert_equal_read_preds(a: dict, b: dict) -> None:
        """
        * Keys are check to be the same at each depth
        * Any ndarray as check for equality with numpy
        * Everything else is checked with regular equality
        """
        # Both arrays should have the same splits
        assert set(a.keys()) == set(b.keys())

        for k in a.keys():
            if isinstance(a[k], dict):
                assert_equal_read_preds(a[k], b[k])
            elif isinstance(a[k], np.ndarray):
                np.testing.assert_array_equal(a[k], b[k])
            else:
                assert a[k] == b[k], f"Key: {k}"

    assert_equal_read_preds(read_preds, ensbuilder.read_preds)
    assert last_hash == ensbuilder.last_hash

    ensemble_memory_file = os.path.join(
        ensemble_backend.internals_directory, "ensemble_read_losses.pkl"
    )
    assert os.path.exists(ensemble_memory_file)

    # Make sure we pickle the correct read scores
    with (open(ensemble_memory_file, "rb")) as memory:
        read_losses = pickle.load(memory)

    assert_equal_read_preds(read_losses, ensbuilder.read_losses)

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
    assert_equal_read_preds(ensbuilder2.read_preds, ensbuilder.read_preds)
    assert_equal_read_preds(ensbuilder2.read_losses, ensbuilder.read_losses)
    assert ensbuilder2.last_hash == ensbuilder.last_hash


@patch("os.path.exists")
@parametrize("metric", [log_loss, accuracy])
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_get_identifiers_from_run_history(
    exists: Mock,
    metric: Scorer,
    ensemble_backend: Backend,
) -> None:
    run_history = RunHistory()
    run_history._add(
        RunKey(
            config_id=3, instance_id='{"task_id": "breast_cancer"}', seed=1, budget=3.0
        ),
        RunValue(
            cost=0.11347517730496459,
            time=0.21858787536621094,
            status=None,
            starttime=time.time(),
            endtime=time.time(),
            additional_info={
                "duration": 0.20323538780212402,
                "num_run": 3,
                "configuration_origin": "Random Search",
            },
        ),
        status=None,
        origin=None,
    )
    run_history._add(
        RunKey(
            config_id=6, instance_id='{"task_id": "breast_cancer"}', seed=1, budget=6.0
        ),
        RunValue(
            cost=2 * 0.11347517730496459,
            time=2 * 0.21858787536621094,
            status=None,
            starttime=time.time(),
            endtime=time.time(),
            additional_info={
                "duration": 0.20323538780212402,
                "num_run": 6,
                "configuration_origin": "Random Search",
            },
        ),
        status=None,
        origin=None,
    )
    return run_history
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


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_ensemble_builder_process_realrun(
    dask_client_single_worker: dask.distributed.Client,
    ensemble_backend: Backend,
) -> None:
    mock_metric = make_scorer("mock", lambda x, y: 0.9)

    manager = EnsembleBuilderManager(
        start_time=time.time(),
        time_left_for_ensembles=1000,
        backend=ensemble_backend,
        dataset_name="Test",
        task=BINARY_CLASSIFICATION,
        metric=mock_metric,
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

    assert "ensemble_optimization_score" in history[0]
    assert history[0]["ensemble_optimization_score"] == 0.9
    assert "ensemble_val_score" in history[0]
    assert history[0]["ensemble_val_score"] == 0.9
    assert "ensemble_test_score" in history[0]
    assert history[0]["ensemble_test_score"] == 0.9


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_ensemble_builder_nbest_remembered(
    ensemble_backend: Backend,
    dask_client_single_worker: dask.distributed.Client,
) -> None:
    """
    Makes sure ensemble builder returns the size of the ensemble that pynisher allowed
    This way, we can remember it and not waste more time trying big ensemble sizes
    """
    manager = EnsembleBuilderManager(
        start_time=time.time(),
        time_left_for_ensembles=1000,
        backend=ensemble_backend,
        dataset_name="Test",
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
    file_path = os.path.join(
        ensemble_backend.internals_directory, "ensemble_read_preds.pkl"
    )
    assert not os.path.exists(file_path)

    manager.build_ensemble(dask_client_single_worker, unit_test=True)

    future = manager.futures[0]
    dask.distributed.wait([future])  # wait for the ensemble process to finish
    assert not os.path.exists(file_path)
    assert future.result() == ([], 2, None, None, None)
