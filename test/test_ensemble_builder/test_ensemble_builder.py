from __future__ import annotations

from typing import Callable

import random
import time
from pathlib import Path

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.ensemble_building import EnsembleBuilder, Run
from autosklearn.metrics import Scorer, accuracy, make_scorer
from autosklearn.util.functional import bound, pairs

import pytest
from pytest_cases import fixture, parametrize
from unittest.mock import Mock, patch

from test.conftest import DEFAULT_SEED
from test.fixtures.metrics import acc_with_X_data


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
    metrics = builder.metrics
    run_okay = [make_run(losses={m.name: 1 for m in metrics}) for _ in range(5)]
    run_modified = [
        make_run(losses={m.name: 1 for m in metrics}, modified=True) for _ in range(5)
    ]

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
    metrics = builder.metrics
    run_okay = [make_run(losses={m.name: 10 for m in metrics}) for _ in range(5)]
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
    metrics = builder.metrics
    dummy = make_run(dummy=True)
    runs = [make_run(losses={m.name: n for m in metrics}) for n in range(10)]

    candidates, discarded = builder.candidate_selection(
        runs,
        dummy,
        better_than_dummy=False,
        nbest=None,
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
    metrics = builder.metrics
    dummy = make_run(dummy=True, losses={m.name: 2 for m in metrics})

    bad_runs = [make_run(predictions=None) for _ in range(5)]

    good_run = make_run(predictions="ensemble", losses={m.name: 1 for m in metrics})

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
    metrics = builder.metrics

    bad_runs = [make_run(losses=None) for _ in range(5)]
    dummy_run = make_run(dummy=True, losses={m.name: 2 for m in metrics})
    good_run = make_run(losses={m.name: 1 for m in metrics})

    runs = bad_runs + [good_run]

    candidates, discarded = builder.candidate_selection(runs, dummy_run)

    assert len(candidates) == 1
    assert len(discarded) == len(bad_runs)

    # Only candidate should be the one with a loss of 1
    assert candidates[0].losses == {m.name: 1 for m in metrics}


def test_candidates_filters_out_better_than_dummy(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * Should filter out runs worse than dummy
    """
    metrics = builder.metrics

    bad_runs = [make_run(losses={m.name: 1 for m in metrics}) for _ in range(2)]
    dummy_run = make_run(dummy=True, losses={m.name: 0 for m in metrics})
    good_runs = [make_run(losses={m.name: -1 for m in metrics}) for _ in range(3)]

    runs = bad_runs + good_runs

    candidates, discarded = builder.candidate_selection(
        runs,
        dummy_run,
        better_than_dummy=True,
    )

    assert set(candidates)

    assert len(candidates) == 3

    for run, metric in zip(candidates, metrics):
        assert run.losses[metric.name] < dummy_run.losses[metric.name]

    assert len(discarded) == 2

    for run, metric in zip(discarded, metrics):
        assert run.losses[metric.name] >= dummy_run.losses[metric.name]


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
    metrics = builder.metrics

    runs = [make_run(losses={m.name: 10 for m in metrics}) for _ in range(10)]
    dummies = [
        make_run(dummy=True, losses={m.name: 0 for m in metrics}) for _ in range(2)
    ]

    candidates, discarded = builder.candidate_selection(
        runs,
        dummies,
        better_than_dummy=True,
    )

    assert len(candidates) == 2
    assert all(run.is_dummy() for run in candidates)


def test_candidates_better_than_dummy_multiobjective(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * For a run to be considered a candidate, it must be better than the best dummy for
      any one of the objectives.
    """
    x = make_scorer("x", lambda: None)
    y = make_scorer("y", lambda: None)

    # The will be one best dummy per objective achieving a loss of 1
    # You can think of this as a joint filter [1, 1] over "x", "y" losses
    dummies = [
        make_run(dummy=True, losses={"x": 1, "y": 500}),
        make_run(dummy=True, losses={"x": 500, "y": 1}),
    ]

    # Clearly better on both objectives
    good_runs = [make_run(losses={"x": 0, "y": 0}) for _ in range(5)]

    # Worse on both
    badd_runs = [make_run(losses={"x": 2, "y": 2}) for _ in range(5)]

    # Better on 1 objective but worse on the other
    may1_runs = [make_run(losses={"x": 2, "y": 0}) for _ in range(5)]
    may2_runs = [make_run(losses={"x": 0, "y": 2}) for _ in range(5)]

    # Better on 1 objective but equal one the other
    may3_runs = [make_run(losses={"x": 1, "y": 0}) for _ in range(5)]
    may4_runs = [make_run(losses={"x": 0, "y": 1}) for _ in range(5)]

    # Equal on 1 objective but worse one the other
    may5_runs = [make_run(losses={"x": 1, "y": 2}) for _ in range(5)]
    may6_runs = [make_run(losses={"x": 2, "y": 1}) for _ in range(5)]

    expected_candidates = [
        *good_runs,
        *may1_runs,
        *may2_runs,
        *may3_runs,
        *may4_runs,
    ]

    expected_discarded = [
        *badd_runs,
        *may5_runs,
        *may6_runs,
    ]

    runs = expected_candidates + expected_discarded

    candidates, discarded = builder.candidate_selection(
        runs, dummies, better_than_dummy=True, metrics=[x, y]
    )

    # No duplicates
    assert len(candidates) == len(set(candidates))

    assert set(candidates) == set(expected_candidates)
    assert discarded == set(expected_discarded)


@parametrize("nbest", [0, 1, 5, 1000])
def test_candidates_nbest_int_single_objective(
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
    # Make sure it's single objective being tested
    assert len(builder.metrics) == 1
    metric = builder.metrics[0]

    n = 10
    expected = int(bound(nbest, bounds=(1, n)))

    dummy = make_run(dummy=True, losses={metric.name: 50_000})
    runs = [make_run(losses={metric.name: i}) for i in range(n)]
    candidates, discarded = builder.candidate_selection(runs, dummy, nbest=nbest)

    assert len(candidates) == expected

    if len(candidates) > 1:
        for a, b in pairs(candidates):
            assert a.losses[metric.name] <= b.losses[metric.name]

    # Make sure all discarded are worse than the worst candidate
    if any(discarded):
        worst_candidate_loss = candidates[-1].losses[metric.name]

        for d in discarded:
            assert worst_candidate_loss <= d.losses[metric.name]


@parametrize("nbest", [0.0, 0.25, 0.5, 1.0])
def test_candidates_nbest_float_single_objective(
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
    # Make sure it's single objective being tested
    assert len(builder.metrics) == 1
    metric = builder.metrics[0]

    n = 10
    expected = int(bound(nbest * n, bounds=(1, n)))

    dummy = make_run(dummy=True, losses={metric.name: 50_000})
    runs = [make_run(id=i, losses={metric.name: i}) for i in range(2, n + 2)]

    candidates, discarded = builder.candidate_selection(runs, dummy, nbest=nbest)

    assert len(candidates) == expected

    if len(candidates) > 1:
        for a, b in pairs(candidates):
            assert a.losses[metric.name] <= b.losses[metric.name]

    # Make sure all discarded are worse than the worst candidate
    if any(discarded):
        worst_candidate_loss = candidates[-1].losses[metric.name]

        for d in discarded:
            assert worst_candidate_loss <= d.losses[metric.name]


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


@parametrize("max_models", [0, 1, 2, 5])
def test_requires_deletion_max_models_multiobjective_no_overlap(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    max_models: int,
) -> None:
    """
    Expects
    -------
    * When deleting runs with multiple objectives, they runs kept should be done so in
      a roundrobin fashion with respect to each metric.
    """
    # In this case, the runs have no overlap as they are round robin'ed, the order they
    # are put in is the same as the order they come out where priority shifts between
    # the three objectives
    runs = [
        make_run(id=2, losses={"x": 0, "y": 100, "z": 100}),
        make_run(id=3, losses={"x": 100, "y": 0, "z": 100}),
        make_run(id=4, losses={"x": 100, "y": 100, "z": 0}),
        #
        make_run(id=5, losses={"x": 1, "y": 100, "z": 100}),
        make_run(id=6, losses={"x": 100, "y": 1, "z": 100}),
        make_run(id=7, losses={"x": 100, "y": 100, "z": 1}),
        #
        make_run(id=8, losses={"x": 2, "y": 100, "z": 100}),
        make_run(id=9, losses={"x": 100, "y": 2, "z": 100}),
        make_run(id=10, losses={"x": 100, "y": 100, "z": 2}),
    ]

    expected_keep = list(runs[:max_models])
    expected_delete = set(runs[max_models:])

    # Dummy metrics, only used for their names
    x = make_scorer("x", lambda: None)
    y = make_scorer("y", lambda: None)
    z = make_scorer("z", lambda: None)

    # Shuffle the runs to ensure correct sorting takes place
    random.shuffle(runs)

    keep, delete = builder.requires_deletion(
        runs,
        max_models=max_models,
        metrics=[x, y, z],
    )

    assert keep == expected_keep
    assert delete == expected_delete


@parametrize("max_models", [0, 1, 2, 5, 7])
def test_requires_deletion_max_models_multiobjective_overlap(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
    max_models: int,
) -> None:
    """
    Expects
    -------
    * When deleting runs with multiple objectives, they runs kept should be done so in
      a roundrobin fashion with respect to each metric. With overlap, it should
      prioritize the first objective.
    """
    # You can read the losses as a binary counter to make sense of it
    runs = [
        make_run(id=2, losses={"x": 0, "y": 0, "z": 0}),
        make_run(id=3, losses={"x": 0, "y": 0, "z": 1}),
        make_run(id=4, losses={"x": 0, "y": 1, "z": 0}),
        make_run(id=5, losses={"x": 0, "y": 1, "z": 1}),
        make_run(id=6, losses={"x": 1, "y": 0, "z": 0}),
        make_run(id=7, losses={"x": 1, "y": 0, "z": 1}),
        make_run(id=8, losses={"x": 1, "y": 1, "z": 0}),
        make_run(id=9, losses={"x": 1, "y": 1, "z": 1}),
    ]

    # The expected order is to prefer the lowest on all objectives
    expected_order = [
        # Best at all
        make_run(id=2, losses={"x": 0, "y": 0, "z": 0}),
        # Best at 2 but prioritizing the first objective
        make_run(id=3, losses={"x": 0, "y": 0, "z": 1}),
        make_run(id=4, losses={"x": 0, "y": 1, "z": 0}),
        make_run(id=6, losses={"x": 1, "y": 0, "z": 0}),
        # Best at 1
        make_run(id=5, losses={"x": 0, "y": 1, "z": 1}),
        make_run(id=7, losses={"x": 1, "y": 0, "z": 1}),
        make_run(id=8, losses={"x": 1, "y": 1, "z": 0}),
        #
        make_run(id=9, losses={"x": 1, "y": 1, "z": 1}),
    ]

    # Dummy metrics, only used for their names
    x = make_scorer("x", lambda: None)
    y = make_scorer("y", lambda: None)

    # Shuffle the runs to ensure correct sorting takes place
    random.shuffle(runs)

    keep, delete = builder.requires_deletion(
        runs,
        max_models=max_models,
        metrics=[x, y],
    )

    assert keep == expected_order[:max_models]
    assert delete == set(expected_order[max_models:])


@parametrize("memory_limit, expected", [(0, 0), (100, 0), (200, 1), (5000, 49)])
def test_requires_memory_limit_single_objective(
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
    # Make sure it's single objective being tested
    assert len(builder.metrics) == 1
    metric = builder.metrics[0]

    runs = [make_run(mem_usage=100, losses={metric.name: -n}) for n in range(50)]
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
        for a, b in pairs(keep):
            assert a.losses[metric.name] <= b.losses[metric.name]

    # Make sure that the best run deleted is still worse than all those kept
    best_deleted = min(r.losses[metric.name] for r in delete)
    assert not any(run.losses[metric.name] > best_deleted for run in keep)


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
    X_data = builder.X_data()
    metric = builder.metrics[0]

    assert builder.loss(run, metric=metric, X_data=X_data, kind=kind) == np.inf


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
    X_data = builder.X_data(kind)
    targets = builder.targets(kind)
    metric = builder.metrics[0]

    run = make_run(predictions={kind: targets})

    assert builder.loss(run, metric=metric, X_data=X_data, kind=kind) < np.inf


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


def test_fit_ensemble_with_no_targets_raises(
    builder: EnsembleBuilder,
    make_run: Callable[..., Run],
) -> None:
    """
    Expects
    -------
    * If no ensemble targets can be found then `fit_ensemble` should fail
    """
    # Delete the targets and then try fit ensemble
    targets_path = Path(builder.backend._get_targets_ensemble_filename())
    targets_path.unlink()

    candidates = [make_run(backend=builder.backend) for _ in range(5)]
    with pytest.raises(ValueError, match="`fit_ensemble` could not find any .*"):
        builder.fit_ensemble(
            candidates=candidates,
            runs=candidates,
        )


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

    ensemble = builder.fit_ensemble(candidates=runs, runs=runs)

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
    X_data = builder.X_data("ensemble")
    targets = builder.targets("ensemble")
    assert targets is not None

    predictions = targets

    runs = [make_run(predictions={"ensemble": predictions}) for _ in range(10)]
    bad_run = make_run(predictions=None)

    runs.append(bad_run)

    with pytest.raises(FileNotFoundError):
        builder.fit_ensemble(candidates=runs, X_data=X_data, targets=targets, runs=runs)


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
    metric = builder.metrics[0]

    # Stick a dummy run and 10 bad runs into the backend
    datamanager = builder.backend.load_datamanager()
    targets = datamanager.data["Y_train"]

    bad_predictions = {"ensemble": np.zeros_like(targets)}
    good_predictions = {"ensemble": targets}

    make_run(dummy=True, losses={metric.name: 10000}, backend=builder.backend)
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


@parametrize("metrics", [accuracy, acc_with_X_data, [accuracy, acc_with_X_data]])
def test_will_build_ensemble_with_different_metrics(
    make_ensemble_builder: Callable[..., EnsembleBuilder],
    make_run: Callable[..., Run],
    metrics: Scorer | list[Scorer],
) -> None:
    """
    Expects
    -------
    * Should be able to build a valid ensemble with different combinations of metrics
    * Should produce a validation score for both "ensemble" and "test" scores
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    builder = make_ensemble_builder(metrics=metrics)

    # Make some runs and stick them in the same backend as the builder
    # Dummy just has a terrible loss for all metrics
    make_run(
        dummy=True,
        losses={m.name: 1000 for m in metrics},
        backend=builder.backend,
    )

    # "Proper" runs will have the correct targets and so be better than dummy
    run_predictions = {
        "ensemble": builder.targets("ensemble"),
        "test": builder.targets("test"),
    }
    for _ in range(5):
        make_run(predictions=run_predictions, backend=builder.backend)

    history, nbest = builder.main()

    # Should only produce one step
    assert len(history) == 1
    hist = history[0]

    # Each of these two keys should be present
    for key in ["ensemble_optimization_score", "ensemble_test_score"]:
        assert key in hist

        # TODO should be updated in next PR
        #   Each of these scores should contain all the metrics
        # for metric in metrics:
        #   assert metric.name in hist[key]


@parametrize("n_least_prioritized", [1, 2, 3, 4])
@parametrize("metrics", [accuracy, acc_with_X_data, [accuracy, acc_with_X_data]])
def test_fit_ensemble_kwargs_priorities(
    make_ensemble_builder: Callable[..., EnsembleBuilder],
    make_run: Callable[..., Run],
    metrics: Scorer | list[Scorer],
    n_least_prioritized: int,
) -> None:
    """
    Expects
    -------
    * Should favour 1) function kwargs, 2) function params 3) init_kwargs 4) init_params
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    class FakeEnsembleClass:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(*args, **kwargs) -> None:
            pass

    # We establish the priorty order and give each one of them a custom metric
    priority = ["function_kwargs", "function_params", "init_kwargs", "init_params"]

    # We reverse the priority and use the `n_least_prioritized` ones
    # with `n_least_prioritized = 3`
    #   reversed =  ["init_params", "init_kwargs", "function_params", "function_kwargs"]
    #   used =      ["init_params", "init_kwargs", "function_params"]
    #   highest =   "function_params"
    reversed_priority = list(reversed(priority))
    used = reversed_priority[:n_least_prioritized]
    highest_priority = used[-1]

    def S(name: str) -> Scorer:
        return make_scorer(name, lambda: None)

    # We now pass in all the places this arguments could be specified
    # Naming them specifically to make it more clear in setup below
    builder_metric = [S("init_params")] if "init_params" in used else None
    fit_ensemble_metric = [S("function_params")] if "function_params" in used else None

    builder_ensemble_kwargs = (
        {"metrics": [S("init_kwargs")]} if "init_kwargs" in used else None
    )
    fit_ensemble_kwargs = (
        {"metrics": [S("function_kwargs")]} if "function_kwargs" in used else None
    )

    builder = make_ensemble_builder(
        metrics=builder_metric,
        ensemble_kwargs=builder_ensemble_kwargs,
    )

    candidates = [make_run() for _ in range(5)]  # Just so something can be run

    ensemble = builder.fit_ensemble(
        metrics=fit_ensemble_metric,
        ensemble_class=FakeEnsembleClass,
        ensemble_kwargs=fit_ensemble_kwargs,
        candidates=candidates,
        runs=candidates,
    )

    # These are the final metrics passed to the ensemble builder when constructed
    passed_metrics = ensemble.kwargs["metrics"]
    metric = passed_metrics[0]

    assert metric.name == highest_priority


@parametrize("metric, should_be_loaded", [(accuracy, False), (acc_with_X_data, True)])
def test_X_data_only_loaded_when_required(
    make_ensemble_builder: Callable[..., EnsembleBuilder],
    make_run: Callable[..., Run],
    metric: Scorer,
    should_be_loaded: bool,
) -> None:
    """
    Expects
    -------
    * Should only load X_train if it's required
    * TODO should only load X_test if it's required
    """
    metrics = [metric]
    builder = make_ensemble_builder(metrics=metrics)

    # Make a dummy which is required for the whole pipeline to run
    make_run(dummy=True, losses={metric.name: 1000}, backend=builder.backend)

    # Make a run that has no losses recorded, forcing us to use the metric
    make_run(
        dummy=False,
        predictions={"ensemble": builder.targets("ensemble")},
        losses=None,
        backend=builder.backend,
    )

    ret_value = builder.X_data()
    builder.X_data = Mock(return_value=ret_value)

    builder.main()

    assert builder.X_data.called == should_be_loaded
