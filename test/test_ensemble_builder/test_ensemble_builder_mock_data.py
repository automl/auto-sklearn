from __future__ import annotations

from typing import Callable

import time
from pathlib import Path

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.ensemble_building import EnsembleBuilder, Run
from autosklearn.metrics import roc_auc

from pytest_cases import fixture, parametrize
from unittest.mock import patch

from test.conftest import DEFAULT_SEED


def test_available_runs(make_ensemble_builder: Callable[..., EnsembleBuilder]) -> None:
    builder = make_ensemble_builder()
    runsdir = Path(builder.backend.get_runs_directory())

    ids = {(0, i, 0.0) for i in range(1, 10)}
    paths = [runsdir / f"{s}_{n}_{b}" for s, n, b in ids]

    for path in paths:
        path.mkdir()

    available_runs = builder.available_runs()

    for run_id in available_runs.keys():
        assert run_id in ids


@parametrize("n_models", [20, 50])
@parametrize("mem_model", [1, 10, 100, 1000])
@parametrize("mem_largest_mult", [1, 2, 10])
@parametrize("n_expected", [1, 3, 5, 10])
@parametrize("largest_is_best", [True, False])
def test_candidates_memory_limit(
    n_models: int,
    mem_model: int,
    mem_largest_mult: int,
    n_expected: int,
    largest_is_best: bool,
    backend: Backend,
    make_ensemble_builder: Callable[..., EnsembleBuilder],
    make_run: Callable[..., Run],
) -> None:
    """
    Parameters
    ----------
    n_models : int
        The number of models to have

    mem_model : int
        The memory consumption per model

    mem_largest_mutl : int
        How much the largest model takes (mem_largest = mem_per_model * mult)

    n_expected : int
        How many models we expect the EnsembleBuilder to save

    largest_is_best: bool
        Whether to include the largest models as one of the best models or as the worst.

    Fixtures
    --------
    make_ensemble_builder: Callable[..., EnsembleBuilder]
    make_run: Callable[..., Run]

    Note
    ----
    We use the parameters here to calculate the `max_models_on_disc` arg to verify
    that with that calculate, we do indeed selected that many models.

        mem_nbest = ... memory of the n best models
        max_models_on_disc = float(mem_nbest + mem_largest_model)

    This is a bit backwards to calculate max_models_on_disc from what we expect but
    it is much easier and still verifies behaviour.

    Expects
    -------
    * The ensemble builder should select the expected number of models given the
      calculated `max_models_on_disc`.
    """
    runs = [
        make_run(id=n, loss=10 * n, mem_usage=mem_model, backend=backend)
        for n in range(1, n_models + 1)
    ]

    mem_largest = mem_model * mem_largest_mult
    if largest_is_best:
        runs[-1]._mem_usage = mem_largest
    else:
        runs[0]._mem_usage = mem_largest

    nbest = sorted(runs, key=lambda run: run.loss)[:n_expected]
    mem_for_nbest = sum(run.mem_usage for run in nbest)
    model_memory_limit = float(mem_for_nbest + mem_largest)  # type: ignore

    builder = make_ensemble_builder(
        max_models_on_disc=model_memory_limit,
        backend=backend,
    )

    candidates, discarded = builder.candidates(
        runs,
        model_memory_limit=model_memory_limit,
    )

    # We expect to save the first n runs as those are the ones with thel lowest loss
    expected = runs[:n_expected]

    assert expected == candidates
    assert set(runs) - set(candidates) == set(discarded)


@parametrize("n_models", [50, 10, 2, 1])
def test_max_models_on_disc_float_always_preserves_best_model(
    n_models: int,
    dummy_backend: Backend,
) -> None:
    """
    Parameters
    ----------
    n_models : int
        The number of models to start with

    Fixtures
    --------
    dummy_backend: Backend
        Just a valid backend, contents don't matter for this test

    Expects
    -------
    * The best model should always be selected even if the memory assigned for models
      on disc does not allow for any models. This is because we need at least one.
    """
    max_models_on_disc = 0.0

    ensbuilder = EnsembleBuilder(
        backend=dummy_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        max_models_on_disc=max_models_on_disc,
        memory_limit=None,
    )

    runs = [
        Run(
            seed=DEFAULT_SEED,
            num_run=n + 1,
            budget=0.0,
            loss=10 * -n,
            loaded=1,
            mem_usage=50 * n,
            ens_file=f"pred{n+1}",
        )
        for n in range(n_models)
    ]
    best_model = min(runs, key=lambda run: run.loss)

    ensbuilder._runs = {run.ens_file: run for run in runs}
    ensbuilder._run_predictions = {
        f"pred{n}": {Y_ENSEMBLE: np.array([1])} for n in range(1, n_models + 1)
    }

    sel_keys = ensbuilder.get_n_best_preds()
    assert [best_model.ens_file] == sel_keys


@parametrize(
    "performance_range_threshold, expected_selected",
    ((0.0, 4), (0.1, 4), (0.3, 3), (0.5, 2), (0.6, 2), (0.8, 1), (1.0, 1)),
)
def test_performance_range_threshold(
    performance_range_threshold: float,
    expected_selected: int,
    dummy_backend: Backend,
) -> None:
    """
    Parameters
    ----------
    performance_range_threshold : float
        THe performance range threshold to use

    expected_selected : int
        The number of selected models for there to be

    Fixtures
    --------
    dummy_backend: Backend
        A valid backend whose contents don't matter for this test

    Expects
    -------
    * Expects the given amount of models to be selected given a performance range
    threshold.
    """
    ensbuilder = EnsembleBuilder(
        backend=dummy_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,
        performance_range_threshold=performance_range_threshold,
    )

    ensbuilder._runs = {
        "A": Run(seed=DEFAULT_SEED, num_run=1, loss=-1, loaded=-1, ens_file=""),
        "B": Run(seed=DEFAULT_SEED, num_run=2, loss=-2, loaded=-1, ens_file=""),
        "C": Run(seed=DEFAULT_SEED, num_run=3, loss=-3, loaded=-1, ens_file=""),
        "D": Run(seed=DEFAULT_SEED, num_run=4, loss=-4, loaded=-1, ens_file=""),
        "E": Run(seed=DEFAULT_SEED, num_run=5, loss=-5, loaded=-1, ens_file=""),
    }
    ensbuilder._run_predictions = {
        name: {preds_key: np.array([1]) for preds_key in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for name in ensbuilder._runs
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
def test_performance_range_threshold_with_ensemble_nbest(
    performance_range_threshold: float,
    ensemble_nbest: int | float,
    expected_selected: int,
    dummy_backend: Backend,
) -> None:
    """
    Parameters
    ----------
    performance_range_threshold : float
    ensemble_nbest : int | float
    expected_selected : int
        The number of models expected to be selected

    Fixtures
    --------
    dummy_backend: Backend
        A backend whose contents are valid and don't matter for this test

    Expects
    -------
    * Given the setup of params for test_performance_range_threshold and ensemble_nbest,
    the expected number of models should be selected.
    """
    ensbuilder = EnsembleBuilder(
        backend=dummy_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,
        ensemble_nbest=ensemble_nbest,
        performance_range_threshold=performance_range_threshold,
        max_models_on_disc=None,
    )
    ensbuilder._runs = {
        "A": Run(seed=DEFAULT_SEED, num_run=1, loss=-1, loaded=-1, ens_file=""),
        "B": Run(seed=DEFAULT_SEED, num_run=2, loss=-2, loaded=-1, ens_file=""),
        "C": Run(seed=DEFAULT_SEED, num_run=3, loss=-3, loaded=-1, ens_file=""),
        "D": Run(seed=DEFAULT_SEED, num_run=4, loss=-4, loaded=-1, ens_file=""),
        "E": Run(seed=DEFAULT_SEED, num_run=5, loss=-5, loaded=-1, ens_file=""),
    }
    ensbuilder._run_predictions = {
        name: {pred_name: np.array([1]) for pred_name in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for name in ensbuilder._runs
    }
    sel_keys = ensbuilder.get_n_best_preds()

    assert len(sel_keys) == expected_selected


@parametrize("time_buffer", [1, 5])
@parametrize("duration", [10, 20])
def test_run_end_at(dummy_backend: Backend, time_buffer: int, duration: int) -> None:
    """
    Parameters
    ----------
    time_buffer: int
        How much time buffer to give to the ensemble builder

    duration: int
        How long to run the ensemble builder for

    Fixtures
    --------
    dummy_backend: Backend
        A valid backend whose contents don't matter for this test

    Expects
    -------
    * The limits enforced by pynisher should account for the time_buffer and duration
      to run for + a little bit of overhead that gets rounded to a second.
    """
    with patch("pynisher.enforce_limits") as pynisher_mock:
        ensbuilder = EnsembleBuilder(
            backend=dummy_backend,
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


def test_can_load_pickled_ndarray_of_dtype_object(dummy_backend: Backend) -> None:
    """
    Fixture
    -------
    dummy_backend: Backend
        A backend with a datamanger so it will load

    Expects
    -------
    * EnsembleBuilder should be able to load np.ndarray's that were saved as a pickled
      object, which happens when the np.ndarray's are of dtype object.

    """
    # TODO Should probably remove this test
    #
    #   I'm not sure why the predictions are stored as pickled objects sometimes
    #   but that's a security vunerability to users using auto-sklearn.
    #
    ensbuilder = EnsembleBuilder(
        backend=dummy_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
    )

    # By specifiyng dtype object, we force it into saving as a pickle
    x = np.array([1, 2, 3, 4], dtype=object)

    path = Path(dummy_backend.internals_directory) / "test.npy"
    with path.open("wb") as f:
        # This is the default value (allow_pickle=True) but we explicitly state it
        np.save(f, x, allow_pickle=True)

    loaded_x = ensbuilder._predictions_from(path)

    np.testing.assert_equal(x, loaded_x)
