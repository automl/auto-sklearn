from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.ensemble_building.builder import Y_TEST, Y_VALID, EnsembleBuilder
from autosklearn.metrics import roc_auc

from pytest_cases import parametrize, parametrize_with_cases
from unittest.mock import Mock, patch

import test.test_ensemble_builder.test_3_models.cases as cases
from test.conftest import DEFAULT_SEED
from test.fixtures.logging import MockLogger


@parametrize_with_cases("ensemble_backend", cases=cases)
def test_read(ensemble_backend: Backend) -> None:
    ensbuilder = EnsembleBuilder(
        backend=ensemble_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
    )

    success = ensbuilder.compute_loss_per_model()
    assert success, f"run_predictions = {str(ensbuilder.run_predictions)}"

    assert len(ensbuilder.run_predictions) == 3, ensbuilder.run_predictions.keys()
    assert len(ensbuilder.run_info) == 3, ensbuilder.run_info.keys()

    runsdir = Path(ensemble_backend.get_runs_directory())
    preds_1 = runsdir / "0_1_0.0" / "predictions_ensemble_0_1_0.0.npy"
    preds_2 = runsdir / "0_2_0.0" / "predictions_ensemble_0_2_0.0.npy"
    preds_3 = runsdir / "0_3_100.0" / "predictions_ensemble_0_3_100.0.npy"

    assert ensbuilder.run_info[str(preds_1)]["ens_loss"] == 0.5
    assert ensbuilder.run_info[str(preds_2)]["ens_loss"] == 0.0
    assert ensbuilder.run_info[str(preds_3)]["ens_loss"] == 0.0


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
@parametrize_with_cases("ensemble_backend", cases=cases)
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

    expected_sel = os.path.join(
        ensemble_backend.temporary_directory,
        ".auto-sklearn/runs/0_2_0.0/predictions_ensemble_0_2_0.0.npy",
    )
    assert sel_keys[0] == expected_sel


@parametrize(
    "max_models_on_disc, expected",
    [
        # If None, no reduction
        (None, 2),
        # If Int, limit only on exceed
        (4, 2),
        (1, 1),
        # If Float, translate float to # models.
        # We mock so sizeof will return 500MB, this means that 500MB is required per run
        # and we also need the 500MB extra as slack room. This means we can't fit 2
        # models in 1499MB but we can in 1500MB. We also don't include the dummy
        # model which explains why even with 9999MB, we still only have 2
        (1499.0, 1),
        (1500.0, 2),
        (9999.0, 2),
    ],
)
@parametrize_with_cases("ensemble_backend", cases=cases)
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

    with patch("autosklearn.ensemble_building.builder.sizeof") as mock:
        mock.return_value = 500

        ensbuilder.compute_loss_per_model()
        sel_keys = ensbuilder.get_n_best_preds()
        assert mock.called
        print(mock.call_args_list)
        assert len(sel_keys) == expected


@parametrize_with_cases("ensemble_backend", cases=cases)
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

    for model in ["0_1_0.0", "0_2_0.0", "0_3_100.0"]:
        filename = os.path.join(
            ensemble_backend.temporary_directory,
            f".auto-sklearn/runs/{model}/predictions_ensemble_{model}.npy",
        )
        ensbuilder.run_info[filename]["ens_loss"] = -1

    sel_keys = ensbuilder.get_n_best_preds()

    best_model = "0_1_0.0"
    expected = os.path.join(
        ensemble_backend.temporary_directory,
        f".auto-sklearn/runs/{best_model}/predictions_ensemble_{best_model}.npy",
    )

    assert len(sel_keys) == 1
    assert sel_keys[0] == expected


@parametrize_with_cases("ensemble_backend", cases=cases)
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
    assert set(ensbuilder.run_predictions.keys()) == set(paths)

    selected = sel_keys
    non_selected = set(paths) - set(sel_keys)

    # not selected --> should still be None
    for key in non_selected:
        assert ensbuilder.run_predictions[key][Y_VALID] is None
        assert ensbuilder.run_predictions[key][Y_TEST] is None

    # selected --> read valid and test predictions
    for key in selected:
        assert ensbuilder.run_predictions[key][Y_VALID] is not None
        assert ensbuilder.run_predictions[key][Y_TEST] is not None


@parametrize_with_cases("ensemble_backend", cases=cases)
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
    y_valid_d2 = ensbuilder.run_predictions[d2][Y_VALID][:, 1]
    np.testing.assert_array_almost_equal(y_valid, y_valid_d2)


@parametrize_with_cases("ensemble_backend", cases=cases)
def test_main(ensemble_backend: Backend) -> None:
    """
    Parameters
    ----------
    ensemble_backend : Backend
        The ensemble_backend to use, this test relies on this specific case

    Expects
    -------
    * There should be "run_predictions" and "run_info" saved to file
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

    assert ensbuilder.run_predictions_path.exists(), list(internals_dir.iterdir())
    assert ensbuilder.run_info_path.exists(), list(internals_dir.iterdir())

    # There should be three preds read
    assert len(ensbuilder.run_predictions) == 3
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


@parametrize_with_cases("ensemble_backend", cases=cases)
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

            assert ensbuilder.run_info_path.exists()
            assert not ensbuilder.run_predictions_path.exists()

            assert mock_logger.warning.call_count == i  # type: ignore

            assert (ensbuilder.ensemble_nbest, ensbuilder.read_at_most) == exp_state

        # At this point, when we've reached (ensemble_nbest, read_at_most) = (0, 1),
        # we can still run the ensbulder but it should just raise an error and not
        # change it's internal state
        ensbuilder.run(time_left=1000, iteration=0, pynisher_context="fork")

        assert ensbuilder.run_info_path.exists()
        assert not ensbuilder.run_predictions_path.exists()

        assert (ensbuilder.ensemble_nbest, ensbuilder.read_at_most) == final_state

        warning_call_count = mock_logger.warning.call_count  # type: ignore
        error_call_count = mock_logger.error.call_count  # type: ignore

        assert warning_call_count == len(intermediate_states)
        assert error_call_count == 1

        for call_arg in mock_logger.error.call_args_list:  # type: ignore
            assert "Memory Exception -- Unable to further reduce" in str(call_arg)
