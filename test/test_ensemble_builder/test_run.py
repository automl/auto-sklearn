from __future__ import annotations

from typing import Callable

import pickle
import time
from pathlib import Path

import numpy as np

from autosklearn.ensemble_building import Run

from pytest_cases import parametrize


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
    * Should update the recorded times so `was_modified` will give False after being
      updated
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

    Note
    ----
    * Not sure this should be supported
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
    make_run: Callable[..., Run], precision: int, expected: type
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

    Note
    ----
    The `time.sleep` here is to give some time between accesses. Using a value of
    `0.01` seemed to be too low for the github action runners
    """
    run = make_run()

    path = run.pred_path()
    before_access = path.stat().st_atime_ns

    time.sleep(1)
    _ = run.predictions()  # Should cache result
    load_access = path.stat().st_atime_ns

    # We test that it was not loaded from disk by checking when it was last accessed
    assert before_access != load_access

    time.sleep(1)
    _ = run.predictions()  # Should use cache result
    cache_access = path.stat().st_atime_ns

    assert cache_access == load_access

    run.unload_cache()

    time.sleep(1)
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


@parametrize(
    "name, expected",
    [
        ("0_0_0.0", True),
        ("1_152_64.24", True),
        ("123412_3462_100.0", True),
        ("tmp_sf12198", False),
        ("tmp_0_0_0.0", False),
    ],
)
def test_valid(name: str, expected: bool) -> None:
    """
    Expects
    -------
    * Should be able to correctly consider valid run dir names
    """
    path = Path(name)
    assert Run.valid(path) == expected
