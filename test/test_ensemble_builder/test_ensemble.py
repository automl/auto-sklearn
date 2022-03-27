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
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.ensemble_builder import (
    Y_ENSEMBLE,
    Y_TEST,
    Y_VALID,
    EnsembleBuilder,
    EnsembleBuilderManager,
)
from autosklearn.metrics import make_scorer, roc_auc

from pytest_cases import fixture, parametrize, parametrize_with_cases
from unittest.mock import Mock, patch

import test.test_ensemble_builder.cases as cases
from test.conftest import DEFAULT_SEED
from test.fixtures.logging import MockLogger


@fixture
def dummy_backend(
    tmp_path: Path,
    make_sklearn_dataset: Callable[..., XYDataManager],
    make_backend: Callable[..., Backend],
) -> Backend:
    datamanager = make_sklearn_dataset(
        name="breast_cancer",
        task=BINARY_CLASSIFICATION,
        feat_type="numerical",  # They're all numerical
        as_datamanager=True,
    )
    backend = make_backend(path=tmp_path / "backend")
    backend.save_datamanager(datamanager)
    return backend


@parametrize("n_models", [20, 50])
@parametrize("mem_model", [1, 10, 100, 1000])
@parametrize("mem_largest_mult", [1, 2, 10])
@parametrize("n_expected", [1, 3, 5, 10])
@parametrize("largest_is_best", [True, False])
def test_max_models_on_disc_with_float_selects_expected_models(
    n_models: int,
    mem_model: int,
    mem_largest_mult: int,
    n_expected: int,
    largest_is_best: bool,
    dummy_backend: Backend
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
    dummy_backend: Backend
        Just a backend that's valid, contents don't matter for this test

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

    # These are arranged so the last one is best, with the lose loss
    losses = [
        {
            "ens_loss": 10 * -n,
            "num_run": n,
            "loaded": 1,
            "seed": DEFAULT_SEED,
            "disc_space_cost_mb": mem_model,
        }
        for n in range(1, n_models + 1)
    ]

    mem_largest = mem_model * mem_largest_mult
    if largest_is_best:
        losses[-1]["disc_space_cost_mb"] = mem_largest
    else:
        losses[0]["disc_space_cost_mb"] = mem_largest

    nbest = sorted(losses, key=lambda item: item["ens_loss"])[:n_expected]
    mem_for_nbest = sum(item["disc_space_cost_mb"] for item in nbest)

    slack = mem_largest  # Slack introduced is the size of the largest model
    max_models_on_disc = float(mem_for_nbest + slack)

    ensbuilder = EnsembleBuilder(
        backend=dummy_backend,
        dataset_name="TEST",
        task_type=BINARY_CLASSIFICATION,
        metric=roc_auc,
        seed=DEFAULT_SEED,  # important to find the test files
        max_models_on_disc=max_models_on_disc,
        memory_limit=None,
    )

    # Enter the models, with each model being progressibly better
    ensbuilder.read_losses = {f"pred{i}": v for i, v in enumerate(losses, start=1)}
    # Make the last model twice as large
    if largest_is_best:
        ensbuilder.read_losses[f"pred{n_models}"]["disc_space_cost_mb"] = mem_largest
    else:
        ensbuilder.read_losses["pred1"]["disc_space_cost_mb"] = mem_largest

    ensbuilder.read_preds = {
        f"pred{n}": {Y_ENSEMBLE: True} for n in range(1, n_models + 1)
    }

    sel_keys = ensbuilder.get_n_best_preds()

    # The last expected_to_save models should be saved, the range iters backwards
    expected = [f"pred{n}" for n in range(n_models, n_models - n_expected, -1)]

    assert len(sel_keys) == len(expected) and sel_keys == expected


@parametrize("n_models", [50, 10, 2, 1])
def test_max_models_on_disc_float_always_preserves_best_model(
    n_models: int,
    dummy_backend: Backend
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

    read_losses = {
        f"pred{n}": {
            "ens_loss": 10 * -n,
            "num_run": n + 1,
            "loaded": 1,
            "seed": DEFAULT_SEED,
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
def test_performance_range_threshold(
    performance_range_threshold: float,
    expected_selected: int,
    dummy_backend: Backend
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
    ensbuilder.read_losses = {
        "A": {"ens_loss": -1, "num_run": 1, "loaded": -1, "seed": DEFAULT_SEED},
        "B": {"ens_loss": -2, "num_run": 2, "loaded": -1, "seed": DEFAULT_SEED},
        "C": {"ens_loss": -3, "num_run": 3, "loaded": -1, "seed": DEFAULT_SEED},
        "D": {"ens_loss": -4, "num_run": 4, "loaded": -1, "seed": DEFAULT_SEED},
        "E": {"ens_loss": -5, "num_run": 5, "loaded": -1, "seed": DEFAULT_SEED},
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
    ensbuilder.read_losses = {
        "A": {"ens_loss": -1, "num_run": 1, "loaded": -1, "seed": DEFAULT_SEED},
        "B": {"ens_loss": -2, "num_run": 2, "loaded": -1, "seed": DEFAULT_SEED},
        "C": {"ens_loss": -3, "num_run": 3, "loaded": -1, "seed": DEFAULT_SEED},
        "D": {"ens_loss": -4, "num_run": 4, "loaded": -1, "seed": DEFAULT_SEED},
        "E": {"ens_loss": -5, "num_run": 5, "loaded": -1, "seed": DEFAULT_SEED},
    }
    ensbuilder.read_preds = {
        name: {pred_name: True for pred_name in (Y_ENSEMBLE, Y_VALID, Y_TEST)}
        for name in ensbuilder.read_losses
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
