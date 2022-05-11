from __future__ import annotations

from typing import Callable

import random
import time
from pathlib import Path

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.ensemble_building import EnsembleBuilder, Run
from autosklearn.util.functional import bound, pairs

import pytest
from pytest_cases import fixture, parametrize
from unittest.mock import patch

from test.conftest import DEFAULT_SEED


@fixture
def builder(make_ensemble_builder: Callable[..., EnsembleBuilder]) -> EnsembleBuilder:
    """A default ensemble builder"""
    return make_ensemble_builder()


@parametrize("kind", ["ensemble", "test"])
def test_targets(builder: EnsembleBuilder, kind: str) -> None:
    """
    Expects
    -------
    * Should be able to load each of the targets
    """
    assert builder.targets(kind) is not None


def test_available_runs(builder: EnsembleBuilder) -> None:
    """
    Expects
    -------
    * Should be able to read runs from the backends rundir where runs are tagged
      {seed}_{numrun}_{budget}
    """
    runsdir = Path(builder.backend.get_runs_directory())

    ids = {(0, i, 0.0) for i in range(1, 10)}
    paths = [runsdir / f"{s}_{n}_{b}" for s, n, b in ids]

    for path in paths:
        path.mkdir()

    available_runs = builder.available_runs()

    assert len(available_runs) == len(ids)
    for run_id in available_runs.keys():
        assert run_id in ids


def test_available_runs_with_bad_dir_contained(builder: EnsembleBuilder) -> None:
    """
    Expects
    -------
    * Should ignore dirs that aren't in format
    """
    runsdir = Path(builder.backend.get_runs_directory())

    ids = {(0, i, 0.0) for i in range(1, 10)}
    paths = [runsdir / f"{s}_{n}_{b}" for s, n, b in ids]

    bad_path = runsdir / "Im_a_bad_path"

    for path in paths + [bad_path]:
        path.mkdir()

    available_runs = builder.available_runs()
    assert len(available_runs) == len(paths)


def test_requires_loss_update_with_modified_runs(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should include runs that were modified, even if they have a loss
    """
    run_okay = [make_run(loss=1) for _ in range(5)]
    run_modified = [make_run(loss=1, modified=True) for _ in range(5)]

    runs = run_okay + run_modified

    requires_update = builder.requires_loss_update(runs)

    assert set(run_modified) == set(requires_update)


def test_requires_loss_update_with_no_loss(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should include runs that have no loss
    """
    run_okay = [make_run(loss=10) for _ in range(5)]
    run_no_loss = [make_run() for _ in range(5)]

    runs = run_okay + run_no_loss

    requires_update = builder.requires_loss_update(runs)

    assert set(run_no_loss) == set(requires_update)


def test_candidates_no_filters(
    builder: EnsembleBuilder, make_run: Callable[..., Run]
) -> None:
    """
    Expects
    -------
    * Should have nothing in common between candidates and discarded
    * Should not filter out any viable runs if no filters set. Here a viable run
      has a loss and ensemble predictions
    """
    dummy = make_run(dummy=True)
    runs = [make_run(loss=n) for n in range(10)]

    candidates, discarded = builder.candidate_selection(
        runs,
        dummy,
        better_than_dummy=False,
        nbest=None,
        performance_range_threshold=None,
    )

    assert len(set(candidates) & discarded) == 0
    assert len(candidates) == len(runs)
    assert len(discarded) == 0


def test_candidates_filters_runs_with_no_predictions(
    builder: EnsembleBuilder, make_run: Callable[..., Run]
) -> None:
    """
    Expects
    -------
    * Should have nothing in common between candidates and discarded
    * Should filter out runs with no "ensemble" predictions
    """
    bad_runs = [make_run(predictions=None) for _ in range(5)]
    dummy = make_run(dummy=True, loss=2)
    good_run = make_run(predictions="ensemble", loss=1)

    runs = bad_runs + [good_run]

    candidates, discarded = builder.candidate_selection(runs, dummy)

    assert len(set(candidates) & discarded) == 0
    assert len(candidates) == 1
    assert len(discarded) == len(bad_runs)
    assert candidates[0].pred_path("ensemble").exists()


def test_candidates_filters_runs_with_no_loss(
    builder: EnsembleBuilder, make_run: Callable[..., Run]
) -> None:
    """
    Expects
    -------
    * Should filter out runs with no loss
    """
    bad_runs = [make_run(loss=None) for _ in range(5)]
    dummy_run = make_run(dummy=True, loss=2)
    good_run = make_run(loss=1)

    runs = bad_runs + [good_run]

    candidates, discarded = builder.candidate_selection(runs, dummy_run)

    assert len(candidates) == 1
    assert len(discarded) == len(bad_runs)
    assert candidates[0].loss == 1


def test_candidates_filters_out_better_than_dummy(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should filter out runs worse than dummy
    """
    bad_runs = [make_run(loss=1) for _ in range(2)]
    dummy_run = make_run(dummy=True, loss=0)
    good_runs = [make_run(loss=-1) for _ in range(3)]

    runs = bad_runs + good_runs

    candidates, discarded = builder.candidate_selection(
        runs, dummy_run, better_than_dummy=True
    )

    assert set(candidates)

    assert len(candidates) == 3
    assert all(run.loss < dummy_run.loss for run in candidates)

    assert len(discarded) == 2
    assert all(run.loss >= dummy_run.loss for run in discarded)


def test_candidates_uses_dummy_if_no_candidates_better(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * If no run is better than a dummy run, the candidates will then consist
      of the dummy runs.
    """
    runs = [make_run(loss=10) for _ in range(10)]
    dummies = [make_run(dummy=True, loss=0) for _ in range(2)]

    candidates, discarded = builder.candidate_selection(
        runs,
        dummies,
        better_than_dummy=True,
    )

    assert len(candidates) == 2
    assert all(run.is_dummy() for run in candidates)


@parametrize("nbest", [0, 1, 5, 1000])
def test_candidates_nbest_int(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    nbest: int,
) -> None:
    """
    Expects
    -------
    * Should only select the nbest candidates
    * They should be ordered by loss
    """
    n = 10
    expected = int(bound(nbest, bounds=(1, n)))

    dummy = make_run(dummy=True)
    runs = [make_run(loss=i) for i in range(n)]
    candidates, discarded = builder.candidate_selection(runs, dummy, nbest=nbest)

    assert len(candidates) == expected

    if len(candidates) > 1:
        assert all(a.loss <= b.loss for a, b in pairs(candidates))

    if any(discarded):
        worst_candidate = candidates[-1]
        assert all(worst_candidate.loss <= d.loss for d in discarded)


@parametrize("nbest", [0.0, 0.25, 0.5, 1.0])
def test_candidates_nbest_float(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    nbest: float,
) -> None:
    """
    Expects
    -------
    * Should select nbest percentage of candidates
    * They should be ordered by loss
    """
    n = 10
    expected = int(bound(nbest * n, bounds=(1, n)))

    dummy = make_run(dummy=True, loss=0)
    runs = [make_run(id=i, loss=i) for i in range(2, n + 2)]
    candidates, discarded = builder.candidate_selection(runs, dummy, nbest=nbest)

    assert len(candidates) == expected

    if len(candidates) > 1:
        assert all(a.loss <= b.loss for a, b in pairs(candidates))

    if any(discarded):
        worst_candidate = candidates[-1]
        assert all(worst_candidate.loss <= d.loss for d in discarded)


@parametrize("threshold", [0.0, 0.25, 0.5, 1.0])
def test_candidates_performance_range_threshold(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    threshold: float,
) -> None:
    """
    Expects
    -------
    * Should select runs that are `threshold` between the dummy loss and the best loss
      This value is captured in `boundary`.
    """
    worst_loss = 100
    best_loss = 0
    dummy_loss = 50

    boundary = threshold * best_loss + (1 - threshold) * dummy_loss

    dummy = make_run(dummy=True, loss=dummy_loss)
    runs = [make_run(loss=loss) for loss in np.linspace(best_loss, worst_loss, 101)]

    candidates, discarded = builder.candidate_selection(
        runs,
        dummy,
        performance_range_threshold=threshold,
    )

    # When no run is better than threshold, we just get 1 candidate,
    # Make sure it's the best
    if len(candidates) == 1:
        assert all(r.loss >= candidates[0].loss for r in discarded)

    else:
        for run in candidates:
            assert run.loss < boundary

        for run in discarded:
            assert run.loss >= boundary


def test_requires_deletion_does_nothing_without_params(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * All runs should be kept
    """
    runs = [make_run() for _ in range(10)]

    keep, delete = builder.requires_deletion(
        runs,
        max_models=None,
        memory_limit=None,
    )

    assert set(runs) == set(keep)
    assert len(delete) == 0


@parametrize("max_models", [0, 1, 2, 5])
def test_requires_deletion_max_models(
    builder: EnsembleBuilder,
    max_models: int,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should keep exactly as many models as `max_models`
    * Should not have any in common between keep and delete
    """
    runs = [make_run() for _ in range(10)]
    keep, delete = builder.requires_deletion(runs=runs, max_models=max_models)

    assert len(keep) == max_models
    assert len(delete) == len(runs) - max_models

    assert not any(set(keep) & set(delete))


@parametrize("memory_limit, expected", [(0, 0), (100, 0), (200, 1), (5000, 49)])
def test_requires_memory_limit(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    memory_limit: int,
    expected: int,
) -> None:
    """
    Expects
    -------
    * Should keep the expected amount of models
    * The kept models should be sorted by lowest loss
    * Should not have any models in common between keep and delete
    * All models kept should be better than those deleted
    """
    runs = [make_run(mem_usage=100, loss=-n) for n in range(50)]
    random.shuffle(runs)

    keep, delete = builder.requires_deletion(runs=runs, memory_limit=memory_limit)

    # The cutoff for memory is (memory_limit - largest)
    # E.g.
    #   5 models at 100 ea = 500mb usage
    #   largest = 100mb
    #   memory_limit = 400mb
    #   cutoff = memory_limit - largest  (400mb - 100mb) = 300mb
    #   We can store 300mb which means the 3 best models
    assert len(keep) == expected
    assert len(delete) == len(runs) - expected

    assert not any(set(keep) & set(delete))

    if len(keep) > 2:
        assert all(a.loss <= b.loss for a, b in pairs(keep))

    best_deleted = min(r.loss for r in delete)
    assert not any(run.loss > best_deleted for run in keep)


@parametrize("kind", ["ensemble", "test"])
def test_loss_with_no_ensemble_targets(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    kind: str,
) -> None:
    """
    Expects
    -------
    * Should give a loss of np.inf if run has no predictions of a given kind
    """
    run = make_run(predictions=None)

    assert builder.loss(run, kind=kind) == np.inf


@parametrize("kind", ["ensemble", "test"])
def test_loss_with_targets(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    kind: str,
) -> None:
    """
    Expects
    -------
    * Should give a loss < np.inf if the predictions exist
    """
    targets = builder.targets(kind)

    run = make_run(predictions={kind: targets})

    assert builder.loss(run, kind) < np.inf


def test_delete_runs(builder: EnsembleBuilder, make_run: Callable[..., Run]) -> None:
    """
    Expects
    -------
    * Should delete runs so they can not be found again by the ensemble builder
    """
    runs = [make_run(backend=builder.backend) for _ in range(5)]
    assert all(run.dir.exists() for run in runs)

    builder.delete_runs(runs)
    assert not any(run.dir.exists() for run in runs)

    loaded = builder.available_runs()
    assert len(loaded) == 0


def test_delete_runs_does_not_delete_dummy(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should
    """
    backend = builder.backend
    normal_runs = [make_run(backend=backend) for _ in range(5)]
    dummy_runs = [make_run(dummy=True, seed=i, backend=backend) for i in range(2)]

    runs = normal_runs + dummy_runs
    assert all(run.dir.exists() for run in runs)

    builder.delete_runs(runs)
    assert not any(run.dir.exists() for run in normal_runs)
    assert all(dummy.dir.exists() for dummy in dummy_runs)

    loaded = builder.available_runs()
    assert set(loaded.values()) == set(dummy_runs)


def test_fit_ensemble_produces_ensemble(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should produce an ensemble if all runs have predictions
    """
    targets = builder.targets("ensemble")
    assert targets is not None

    predictions = targets
    runs = [make_run(predictions={"ensemble": predictions}) for _ in range(10)]

    ensemble = builder.fit_ensemble(runs, targets)

    assert ensemble is not None


def test_fit_with_error_gives_no_ensemble(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * A run without predictions will raise an error will cause `fit_ensemble` to fail
      as it requires all runs to have valid predictions
    """
    targets = builder.targets("ensemble")
    assert targets is not None

    predictions = targets

    runs = [make_run(predictions={"ensemble": predictions}) for _ in range(10)]
    bad_run = make_run(predictions=None)

    runs.append(bad_run)

    with pytest.raises(FileNotFoundError):
        builder.fit_ensemble(runs, targets)


@parametrize("time_buffer", [1, 5])
@parametrize("duration", [10, 20])
def test_run_end_at(builder: EnsembleBuilder, time_buffer: int, duration: int) -> None:
    """
    Expects
    -------
    * The limits enforced by pynisher should account for the time_buffer and duration
      to run for + a little bit of overhead that gets rounded to a second.
    """
    with patch("pynisher.enforce_limits") as pynisher_mock:
        builder.run(
            end_at=time.time() + duration,
            iteration=1,
            time_buffer=time_buffer,
            pynisher_context="forkserver",
        )
        # The 1 comes from the small overhead in conjuction with rounding down
        expected = duration - time_buffer - 1

        # The 1 comes from the small overhead in conjuction with rounding down
        expected = duration - time_buffer - 1
        assert pynisher_mock.call_args_list[0][1]["wall_time_in_s"] == expected


def test_deletion_will_not_break_current_ensemble(
    make_backend: Callable[..., Backend],
    make_ensemble_builder: Callable[..., EnsembleBuilder],
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * When running the builder, it's previous ensemble should not have it's runs deleted
      until a new ensemble is built.
    """
    # Make a builder with this backend and limit it to only allow 10 models on disc
    builder = make_ensemble_builder(
        max_models_on_disc=10,
        seed=DEFAULT_SEED,
    )

    # Stick a dummy run and 10 bad runs into the backend
    datamanager = builder.backend.load_datamanager()
    targets = datamanager.data["Y_train"]

    bad_predictions = {"ensemble": np.zeros_like(targets)}
    good_predictions = {"ensemble": targets}

    make_run(dummy=True, loss=10000, backend=builder.backend)
    bad_runs = [
        make_run(backend=builder.backend, predictions=bad_predictions)
        for _ in range(10)
    ]

    ens_dir = Path(builder.backend.get_ensemble_dir())

    # Make sure there's no ensemble and run with the candidates available
    assert not ens_dir.exists()
    builder.main(time_left=100)

    # Make sure an ensemble was built
    assert ens_dir.exists()
    first_builder_contents = set(ens_dir.iterdir())

    # Create 10 new and better runs and put them in the backend
    new_runs = [
        make_run(backend=builder.backend, predictions=good_predictions)
        for _ in range(10)
    ]

    # Now we make `save_ensemble` crash so that even though we run the builder, it does
    # not manage to save the new ensemble
    with patch.object(builder.backend, "save_ensemble", side_effect=ValueError):
        try:
            builder.main(time_left=100)
        except Exception:
            pass

    # Ensure that no new ensemble was created
    second_builder_contents = set(ens_dir.iterdir())
    assert first_builder_contents == second_builder_contents

    # Now we make sure that the ensemble there still has access to all the bad models
    # that it contained from the first run, even though the second crashed.
    available_runs = builder.available_runs().values()
    for run in bad_runs + new_runs:
        assert run in available_runs

    # As a sanity check, run the builder one more time without crashing and make
    # sure the bad runs are removed with the good ones kept.
    # We remove its previous candidates so that it won't remember previous candidates
    # and will fit a new ensemble
    builder.previous_candidates_path.unlink()
    builder.main(time_left=100)
    available_runs = builder.available_runs().values()

    for run in bad_runs:
        assert run not in available_runs

    for run in new_runs:
        assert run in available_runs
