"""
This file tests the ensemble builder with real runs generated from running AutoML
"""
from __future__ import annotations

from typing import Callable

from autosklearn.automl import AutoML
from autosklearn.ensemble_building.builder import EnsembleBuilder

import pytest
from pytest_cases import parametrize_with_cases
from pytest_cases.filters import has_tag
from unittest.mock import MagicMock, patch

import test.test_automl.cases as cases


@parametrize_with_cases(
    "automl",
    cases=cases,
    filter=has_tag("fitted") & ~has_tag("no_ensemble"),
)
def case_real_runs(
    automl: AutoML,
    make_ensemble_builder: Callable[..., EnsembleBuilder],
) -> EnsembleBuilder:
    """Uses real runs from a fitted automl instance which have an ensemble

    This will copy the ensemble builder based on the AutoML instance parameterss. This
    includes ensemble_nbest, ensemble_size, etc...
    """
    builder = make_ensemble_builder(automl=automl)
    return builder


@parametrize_with_cases("builder", cases=case_real_runs)
def test_run_builds_valid_ensemble(builder: EnsembleBuilder) -> None:
    """
    Expects
    -------
    * Using the same builder as used in the real run should result in the same
      candidate models for the ensemble.
    * Check that there is no overlap between candidate models and those deleted
    * The generated ensemble should not be empty
    * If any deleted, should be no overlap with those deleted and those in ensemble
    * If any deleted, they should all be worse than those in the ensemble
    """
    # We need to clear previous candidates so the ensemble builder is presented with
    # only "new" runs and has no information of previous candidates
    if builder.previous_candidates_path.exists():
        builder.previous_candidates_path.unlink()

    # So we can capture the saved ensemble
    mock_save = MagicMock()
    builder.backend.save_ensemble = mock_save  # type: ignore

    # So we can capture what was deleted
    mock_delete = MagicMock()
    builder.delete_runs = mock_delete  # type: ignore

    # So we can capture the candidate runs used, we still wrap the actual fitting
    with patch.object(builder, "fit_ensemble", wraps=builder.fit_ensemble) as mock_fit:
        history, nbest = builder.main()

    # Check the ensemble was fitted once
    mock_save.assert_called_once()
    _, kwargs = mock_save.call_args
    ens = kwargs["ensemble"]  # `backend.save_ensemble(ens, ...)`
    ensemble_ids = set(ens.get_selected_model_identifiers())
    assert len(ensemble_ids) > 0

    # Check that the ids of runs in the ensemble were all candidates
    candidates = mock_fit.call_args[0][0]  # `fit_ensemble(candidates, ...)`
    candidate_ids = {run.id for run in candidates}
    assert ensemble_ids <= candidate_ids

    args, _ = mock_delete.call_args
    deleted = args[0]  # `delete_runs(runs)`

    # If we deleted runs, we better make sure of a few things
    if len(deleted) > 0:
        deleted_ids = {run.id for run in deleted}

        # Make sure theres no overlap between candidates/ensemble and those deleted
        assert not any(deleted_ids & candidate_ids)
        assert not any(deleted_ids & ensemble_ids)

        # Make sure that the best deleted model is better than the worst candidate
        best_deleted = min(deleted, key=lambda r: (r.loss, r.num_run))
        worst_candidate = max(candidates, key=lambda r: (r.loss, r.num_run))

        a = (worst_candidate.loss, worst_candidate.num_run)
        b = (best_deleted.loss, best_deleted.num_run)
        assert a <= b


@parametrize_with_cases("builder", cases=case_real_runs)
def test_does_not_update_ensemble_with_no_new_runs(builder: EnsembleBuilder) -> None:
    """
    Expects
    -------
    * No new ensemble should be fitted with no new runs and no runs updated.
      Since this is from a real AutoML run, running the builder again without having
      trained any new models should mean that the `fit_ensemble` is never run in the
      EnsembleBuilder.
    """
    if not builder.previous_candidates_path.exists():
        pytest.skip("Test only valid when builder has previous candidates")

    prev_history = builder.ensemble_history
    prev_nbest = builder.ensemble_nbest

    # So we can wrap and test if fit ensemble gets called
    with patch.object(builder, "fit_ensemble", wraps=builder.fit_ensemble) as mock_fit:
        history, nbest = builder.main()

    assert history == prev_history
    assert prev_nbest == nbest
    assert mock_fit.call_count == 0
