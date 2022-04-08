from __future__ import annotations

from typing import Callable

import math
import pickle
import time
from pathlib import Path

import numpy as np

from autosklearn.ensemble_building.run import Run

from pytest_cases import fixture, parametrize

from test.conftest import DEFAULT_SEED


@fixture
def make_run(tmp_path: Path) -> Callable[..., Run]:
    def _make(
        id: int | None = 2,
        seed: int = DEFAULT_SEED,
        budget: float = 0.0,
        loss: float | None = None,
        model_size: int | None = None,
        predictions: list[str] | dict[str, np.ndarray] | None = None,
    ) -> Run:
        model_id = f"{seed}_{id}_{budget}"
        dir = tmp_path / model_id

        if not dir.exists():
            dir.mkdir()

        # Populate if None
        if predictions is None:
            predictions = ["ensemble", "valid", "test"]

        # Convert to dict
        if isinstance(predictions, list):
            dummy = np.asarray([[0]])
            predictions = {kind: dummy for kind in predictions}

        # Write them
        if isinstance(predictions, dict):
            for kind, val in predictions.items():
                fname = f"predictions_{kind}_{seed}_{id}_{budget}.npy"
                with (dir / fname).open("wb") as f:
                    np.save(f, val)

        run = Run(dir)

        if loss is not None:
            run.loss = loss

        # MB
        if model_size is not None:
            n_bytes = int(model_size * math.pow(1024, 2))
            model_path = dir / f"{seed}.{id}.{budget}.model"
            with model_path.open("wb") as f:
                f.write(bytearray(n_bytes))

        return run

    return _make


def test_is_dummy(make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * We expect runs with an num_run (id) of 1 to be a dummy
    """
    run = make_run(id=1)
    assert run.is_dummy()

    run = make_run(id=2)
    assert not run.is_dummy()


def test_was_modified(make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * Should properly indicate when a file was modified
    """
    run = make_run()
    assert not run.was_modified()

    time.sleep(0.2)  # Just to give some time after creation
    path = run.pred_path("ensemble")
    path.touch()

    assert run.was_modified()


def test_record_modified_times_with_was_modified(make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * Updating the recorded times should not trigger `was_modified`
    * Should update the recorded times so `was_modified` will give False after being updated
    """
    run = make_run()
    path = run.pred_path("ensemble")

    time.sleep(0.2)
    run.record_modified_times()
    assert not run.was_modified()

    time.sleep(0.2)
    path.touch()
    assert run.was_modified()

    time.sleep(0.2)
    run.record_modified_times()
    assert not run.was_modified()


def test_predictions_pickled(make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * Should be able to load pickled predictions
    """
    run = make_run(predictions=[])
    x = np.array([0])

    path = run.pred_path("ensemble")
    with path.open("wb") as f:
        pickle.dump(x, f)

    assert run.predictions("ensemble") is not None


@parametrize(
    "precision, expected", [(16, np.float16), (32, np.float32), (64, np.float64)]
)
def test_predictions_precision(
    make_run: Callable[..., Run],
    precision: int,
    expected: type
) -> None:
    """
    Expects
    -------
    * Loading predictions with a given precision should load the expected type
    """
    run = make_run()
    assert run.predictions(precision=precision).dtype == expected


def test_caching(make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * Attempting to load the same predictions again will cause the result to be cached
    * Unloading the cache will cause it to reload and reread the predictions
    """
    run = make_run()

    path = run.pred_path()
    before_access = path.stat().st_atime_ns

    time.sleep(0.01)
    _ = run.predictions()  # Should cache result
    load_access = path.stat().st_atime_ns

    # We test that it was not loaded from disk by checking when it was last accessed
    assert before_access != load_access

    time.sleep(0.01)
    _ = run.predictions()  # Should use cache result
    cache_access = path.stat().st_atime_ns

    assert cache_access == load_access

    run.unload_cache()

    time.sleep(0.01)
    _ = run.predictions()  # Should have reloaded it
    reloaded_access = path.stat().st_atime_ns

    assert reloaded_access != cache_access


def test_equality(make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * Two runs with the same id's should be considered equal
    * Otherwise, they should be considered different
    """
    r1 = make_run(id=1, budget=49.3, seed=3)
    r2 = make_run(id=1, budget=49.3, seed=3)

    assert r1 == r2

    r3 = make_run(id=1, budget=0.0, seed=3)

    assert r1 != r3
    assert r2 != r3
