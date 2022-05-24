"""
This file tests the ensemble builder with real runs generated from running AutoML
"""
from __future__ import annotations

from typing import Callable

from autosklearn.automl import AutoML
from autosklearn.ensemble_building.builder import EnsembleBuilder

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
    with patch(
        "autosklearn.ensembles.ensemble_selection.EnsembleSelection"
        ".get_validation_performance"
    ) as mock_get_validation_performance:
        mock_get_validation_performance.return_value = 0.2
        with patch.object(
            builder, "fit_ensemble", wraps=builder.fit_ensemble
        ) as mock_fit:
            history, nbest = builder.main()

    # Check the ensemble was fitted once
    mock_save.assert_called_once()
    _, kwargs = mock_save.call_args
    ens = kwargs["ensemble"]  # `backend.save_ensemble(ens, ...)`
    ensemble_ids = set(ens.get_selected_model_identifiers())
    assert len(ensemble_ids) > 0

    assert mock_fit.call_count == 1
    # Check that the ids of runs in the ensemble were all candidates
    candidates = mock_fit.call_args[1]["candidates"]
    candidate_ids = {run.id for run in candidates}
    assert ensemble_ids <= candidate_ids

    # Could be the case no run is deleted
    if not mock_delete.called:
        return

    args, _ = mock_delete.call_args
    deleted = args[0]  # `delete_runs(runs)`

    # If we deleted runs, we better make sure of a few things
    if len(deleted) > 0:
        deleted_ids = {run.id for run in deleted}

        # Make sure theres no overlap between candidates/ensemble and those deleted
        assert not any(deleted_ids & candidate_ids)
        assert not any(deleted_ids & ensemble_ids)

        # Make sure that the best deleted model is still worse than the worst candidate
        # This does not necessarily hold with respect to any single metric in the
        # multiobjective case as the best deleted may have been deleted with respect to
        # some other metric when compared to the worst candidate, we can't know which.
        if len(builder.metrics) == 1:
            metric = builder.metrics[0]
            best_deleted = min(
                deleted, key=lambda r: (r.losses[metric.name], r.num_run)
            )
            worst_candidate = max(
                candidates, key=lambda r: (r.losses[metric.name], r.num_run)
            )

            a = (worst_candidate.losses[metric.name], worst_candidate.num_run)
            b = (best_deleted.losses[metric.name], best_deleted.num_run)
            assert a <= b
