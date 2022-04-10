from __future__ import annotations

from typing import Callable

from pathlib import Path

import numpy as np

from autosklearn.ensemble_building import EnsembleBuilder, Run
from autosklearn.util.functional import bound, pairs

from pytest_cases import fixture, parametrize


@fixture
def builder(make_ensemble_builder: Callable[..., EnsembleBuilder]) -> EnsembleBuilder:
    return make_ensemble_builder()


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

    assert len(candidates) == len(runs)
    assert len(discarded) == 0


def test_candidates_filters_runs_with_no_predictions(
    builder: EnsembleBuilder, make_run: Callable[..., Run]
) -> None:
    """
    Expects
    -------
    * Should filter out runs with no "ensemble" predictions
    """
    bad_runs = [make_run(predictions=None) for _ in range(5)]
    dummy = make_run(dummy=True, loss=2)
    good_run = make_run(predictions="ensemble", loss=1)

    runs = bad_runs + [good_run]

    candidates, discarded = builder.candidate_selection(runs, dummy)

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
