from __future__ import annotations

from typing import Callable

import os
import pickle
import time
from pathlib import Path

import dask.distributed
import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.ensemble_builder import (
    Y_ENSEMBLE,
    Y_TEST,
    Y_VALID,
    EnsembleBuilder,
    EnsembleBuilderManager,
)
from autosklearn.metrics import make_scorer, roc_auc

from pytest_cases import parametrize, parametrize_with_cases
from unittest.mock import Mock, patch

import test.test_ensemble_builder.cases as cases
from test.conftest import DEFAULT_SEED
from test.fixtures.logging import MockLogger

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
    expected = os.path.join(
        ensemble_backend.temporary_directory,
        f".auto-sklearn/runs/{best_model}/predictions_ensemble_{best_model}.npy",
    )

    assert len(sel_keys) == 1
    assert sel_keys[0] == expected


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
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The ensemble_backend to use, this test relies on this specific case

    Expects
    -------
    * There should be "read_preds" and "read_losses" saved to file
    * There should be 3 model reads
    * There should be a hash for the preds read in
    * The true targets should have been read in
    * The length of the history returned by run should be the same as the iterations
      performed.
    * The run history should contain "optimization", "val" and "test" scores, each being
      the same at 1.0 due to the setup of "setup_3_models".
    """
    iters = 1

    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
    )

    run_history, ensemble_nbest, _, _, _ = ensbuilder.main(
        time_left=np.inf,
        iteration=iters,
        return_predictions=False,
    )

    internals_dir = Path(ensemble_backend.internals_directory)
    read_preds_path = internals_dir / "ensemble_read_preds.pkl"
    read_losses_path = internals_dir / "ensemble_read_losses.pkl"

    assert read_preds_path.exists(), list(internals_dir.iterdir())
    assert read_losses_path.exists(), list(internals_dir.iterdir())

    # There should be three preds read
    assert len(ensbuilder.read_preds) == 3
    assert ensbuilder.last_hash is not None
    assert ensbuilder.y_true_ensemble is not None

    # We expect as many iterations as the iters param
    assert len(run_history) == iters
    hist_item = run_history[0]

    # As the data loader loads the same val/train/test
    # we expect 1.0 as score and all keys available
    expected_performance = {
        "ensemble_val_score": 1.0,
        "ensemble_test_score": 1.0,
        "ensemble_optimization_score": 1.0,
    }

    assert all(key in hist_item for key in expected_performance)
    assert all(hist_item[key] == score for key, score in expected_performance.items())
    assert "Timestamp" in hist_item


@parametrize("time_buffer", [1, 5])
@parametrize("duration", [10, 20])
@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_run_end_at(ensemble_backend: Backend, time_buffer: int, duration: int) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The backend to use

    time_buffer: int
        How much time buffer to give to the ensemble builder

    duration: int
        How long to run the ensemble builder for

    Expects
    -------
    * The limits enforced by pynisher should account for the time_buffer and duration
      to run for + a little bit of overhead that gets rounded to a second.
    """
    with patch("pynisher.enforce_limits") as pynisher_mock:
        ensbuilder = EnsembleBuilder(
            backend=ensemble_backend,
            dataset_name="TEST",
            task_type=BINARY_CLASSIFICATION,
            metric=roc_auc,
        )

        ensbuilder.run(
            end_at=time.time() + duration,
            iteration=1,
            time_buffer=time_buffer,
            pynisher_context="forkserver",
        )

        # The 1 comes from the small overhead in conjuction with rounding down
        expected = duration - time_buffer - 1
        assert pynisher_mock.call_args_list[0][1]["wall_time_in_s"] == expected


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_limit(
    ensemble_backend: Backend,
    mock_logger: MockLogger,
) -> None:
    """

    Parameters
    ----------
    ensemble_backend : Backend
        The backend setup to use

    Fixtures
    --------
    mock_logger: MockLogger
        A logger to inject into the EnsembleBuilder for tracking calls

    Expects
    -------
    * Running from (ensemble_nbest, read_at_most) = (10, 5) where a memory exception
      occurs in each run, we expect ensemble_nbest to be halved continuously until
      it reaches 0, at which point read_at_most is reduced directly to 1.
    """
    expected_states = [(10, 5), (5, 5), (2, 5), (1, 5), (0, 1)]

    starting_state = expected_states[0]
    intermediate_states = expected_states[1:-1]
    final_state = expected_states[-1]

    starting_nbest, starting_read_at_most = starting_state

    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        ensemble_nbest=starting_nbest,
        read_at_most=starting_read_at_most,
        memory_limit=1,
    )

    # Force a memory error to occur
    ensbuilder.predict = Mock(side_effect=MemoryError)  # type: ignore
    ensbuilder.logger = mock_logger  # Mock its logger
    ensbuilder.SAVE2DISC = False

    internal_dir = Path(ensemble_backend.internals_directory)
    read_losses_file = internal_dir / "ensemble_read_losses.pkl"
    read_preds_file = internal_dir / "ensemble_read_preds.pkl"

    def mtime_mock(filename: str) -> float:
        """TODO, not really sure why we have to force these"""
        path = Path(filename)
        mtimes = {
            # At second 0
            "predictions_ensemble_0_1_0.0.npy": 0.0,
            "predictions_valid_0_1_0.0.npy": 0.1,
            "predictions_test_0_1_0.0.npy": 0.2,
            # At second 1
            "predictions_ensemble_0_2_0.0.npy": 1.0,
            "predictions_valid_0_2_0.0.npy": 1.1,
            "predictions_test_0_2_0.0.npy": 1.2,
            # At second 2
            "predictions_ensemble_0_3_100.0.npy": 2.0,
            "predictions_valid_0_3_100.0.npy": 2.1,
            "predictions_test_0_3_100.0.npy": 2.2,
        }
        return mtimes[path.name]

    with patch("os.path.getmtime") as mtime:
        mtime.side_effect = mtime_mock

        starting_state = (starting_nbest, starting_read_at_most)
        assert (ensbuilder.ensemble_nbest, ensbuilder.read_at_most) == starting_state

        intermediate_states = [(5, 5), (2, 5), (1, 5), (0, 1)]
        for i, exp_state in enumerate(intermediate_states, start=1):
            ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")

            assert read_losses_file.exists()
            assert not read_preds_file.exists()

            assert mock_logger.warning.call_count == i  # type: ignore

            assert (ensbuilder.ensemble_nbest, ensbuilder.read_at_most) == exp_state

        # At this point, when we've reached (ensemble_nbest, read_at_most) = (0, 1),
        # we can still run the ensbulder but it should just raise an error and not
        # change it's internal state
        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")

        assert read_losses_file.exists()
        assert not read_preds_file.exists()

        assert (ensbuilder.ensemble_nbest, ensbuilder.read_at_most) == final_state

        warning_call_count = mock_logger.warning.call_count  # type: ignore
        error_call_count = mock_logger.error.call_count  # type: ignore

        assert warning_call_count == len(intermediate_states)
        assert error_call_count == 1

        for call_arg in mock_logger.error.call_args_list:  # type: ignore
            assert "Memory Exception -- Unable to further reduce" in str(call_arg)


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_read_pickle_read_preds(ensemble_backend: Backend) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        THe ensemble backend to use

    Expects
    -------
    * The read_losses and read_preds should be cached between creation of
      the EnsembleBuilder.
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
    internal_dir = Path(ensemble_backend.internals_directory)
    losses_file = internal_dir / "ensemble_read_losses.pkl"
    memory_file = internal_dir / "ensemble_read_preds.pkl"

    assert memory_file.exists()

    # Make sure we pickle the correct read preads and hash
    with memory_file.open("rb") as memory:
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

    assert losses_file.exists()

    # Make sure we pickle the correct read scores
    with losses_file.open("rb") as memory:
        read_losses = pickle.load(memory)

    assert_equal_read_preds(read_losses, ensbuilder.read_losses)

    # Then create a new instance, which should automatically read this file
    ensbuilder2 = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=0,  # important to find the test files
        ensemble_nbest=2,
        max_models_on_disc=None,
    )
    assert_equal_read_preds(ensbuilder2.read_preds, ensbuilder.read_preds)
    assert_equal_read_preds(ensbuilder2.read_losses, ensbuilder.read_losses)
    assert ensbuilder2.last_hash == ensbuilder.last_hash


@parametrize_with_cases("ensemble_backend", cases=cases, has_tag=["setup_3_models"])
def test_ensemble_builder_process_realrun(
    ensemble_backend: Backend,
    make_dask_client: Callable[..., dask.distributed.Client],
) -> None:
    """

    Parameters
    ----------
    ensemble_backend : Backend
        The backend to use, doesn't really matter which kind

    Fixtures
    --------
    make_dask_client : Callable[..., [dask.distributed.Client]]

    Expects
    -------
    * With 1 iteration, the history should only be of length one
    * The expected ensmble score keys for "optimization", "valid" and "test" should
      be in the one history item.
    * The "Timestamp" key should be in the history item
    * With a metric that always returns 0.9, each ensemble score should be 0.9 in the
      history item
    """
    dask_client = make_dask_client(n_workers=1)
    mock_metric = make_scorer("mock", lambda x, y: 0.9)
    iterations = 1

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
        seed=DEFAULT_SEED,
        precision=32,
        max_iterations=iterations,
        read_at_most=np.inf,
        ensemble_memory_limit=None,
        random_state=0,
    )
    manager.build_ensemble(dask_client)
    future = manager.futures.pop()
    dask.distributed.wait([future])  # wait for the ensemble process to finish

    result = future.result()
    history, _, _, _, _ = result

    assert len(history) == iterations

    hist_item = history[0]

    expected_scores = {
        f"ensemble_{key}_score": 0.9 for key in ["optimization", "val", "test"]
    }

    assert "Timestamp" in hist_item
    assert all(key in hist_item for key in expected_scores)
    assert all(hist_item[key] == expected_scores[key] for key in expected_scores)
