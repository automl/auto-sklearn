from __future__ import annotations

from typing import Callable

from autosklearn.automl import AutoML
from autosklearn.ensemble_building.builder import EnsembleBuilder

from pytest_cases import parametrize_with_cases
from unittest.mock import MagicMock, patch

import test.test_automl.cases as cases
from test.conftest import DEFAULT_SEED


@parametrize_with_cases("automl", cases=cases, has_tag="fitted")
def case_real_runs(
    automl: AutoML,
    make_ensemble_builder: Callable[..., EnsembleBuilder],
) -> EnsembleBuilder:
    """Uses real runs from a fitted automl instance"""
    builder = make_ensemble_builder(
        backend=automl._backend,
        metric=automl._metric,
        task_type=automl._task,
        dataset_name=automl._dataset_name,
        seed=automl._seed,
        logger_port=automl._logger_port,
        random_state=DEFAULT_SEED,
    )
    return builder


@parametrize_with_cases("builder", cases=case_real_runs)
def test_run_builds_valid_ensemble(builder: EnsembleBuilder) -> None:
    """
    Expects
    -------
    * The history returned should not be empty
    * The generated ensemble should not be empty
    * If any deleted, should be no overlap with those deleted and ensemble
    * If any deleted, they should all be worse than those in the ensemble
    """
    # So we can capture the saved ensemble
    mock_save = MagicMock()
    builder.backend.save_ensemble = mock_save  # type: ignore

    # So we can capture what was deleted
    mock_delete = MagicMock()
    builder.delete_runs = mock_delete  # type: ignore

    # So we can capture the candidate runs used, we still wrap the actual fitting
    with patch.object(builder, "fit_ensemble", wraps=builder.fit_ensemble) as mock_fit:
        history, nbest = builder.main()

    assert history is not None

    ens, _, _ = mock_save.call_args[0]
    assert len(ens.get_selected_model_identifiers()) > 0

    ens_ids = set(ens.get_selected_model_identifiers())
    deleted = mock_delete.call_args[0][0]

    # If we deleted runs, we better make sure they're worse than what's
    # in the ensemble
    if len(deleted) > 0:
        deleted_ids = {run.id for run in deleted}
        assert len(ens_ids & deleted_ids) == 0

        ensemble_candidates = mock_fit.call_args[0][0]

        best_deleted = min(deleted, key=lambda r: (r.loss, r.num_run))
        worst_candidate = max(ensemble_candidates, key=lambda r: (r.loss, r.num_run))

        a = (worst_candidate.loss, worst_candidate.num_run)
        b = (best_deleted.loss, best_deleted.num_run)
        assert a <= b


@parametrize_with_cases("builder", cases=case_real_runs)
def test_main(builder: EnsembleBuilder) -> None:
    result = builder.run(1, time_left=10)
    raise ValueError(x)
