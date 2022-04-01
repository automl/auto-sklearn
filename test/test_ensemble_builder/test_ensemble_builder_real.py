from __future__ import annotations

from typing import Any, Callable

from pathlib import Path
from shutil import rmtree

from autosklearn.automl import AutoML
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.ensemble_building.builder import EnsembleBuilder

from pytest_cases import parametrize_with_cases

import test.test_automl.cases as cases


@parametrize_with_cases("automl", cases=cases, has_tag="fitted")
def case_ensemble_builder_with_real_runs(
    tmp_path: Path,
    automl: AutoML,
    make_backend: Callable[..., Backend],
) -> tuple[Backend, dict[str, Any]]:
    """Gives the backend for from the cached automl instance in `test_automl/cases.py`

    We do this by copying the backend produced from these cached automl runs to a new
    tmp directory for the ensemble builder tests to run from

    We also delete ensemble building specific things so that ensemble sees them as
    just a collection of runs and no previous ensemble building has been done.
    """
    original_backend = automl._backend
    backend_path = tmp_path / "backend"

    backend = make_backend(path=backend_path, template=original_backend)
    assert backend.internals_directory != original_backend.internals_directory

    ensemble_dir = Path(backend.get_ensemble_dir())
    if ensemble_dir.exists():
        rmtree(ensemble_dir)

    ensemble_hist = Path(backend.internals_directory) / "ensemble_history.json"
    if ensemble_hist.exists():
        ensemble_hist.unlink()

    # This is extra information required to build the ensemble builder exactly
    # as was created by the AutoML object
    builder = EnsembleBuilder(
        backend=backend,
        dataset_name=automl._dataset_name,  # type: ignore is not None
        task_type=automl._task,  # type: ignore is not None
        metric=automl._metric,  # type: ignore is not None
        seed=automl._seed,
        max_models_on_disc=automl._max_models_on_disc,
        memory_limit=automl._memory_limit,
    )
    return builder


@parametrize_with_cases("builder", cases=case_ensemble_builder_with_real_runs)
def test_outputs(builder: EnsembleBuilder) -> None:
    """
    Fixtures
    --------
    builder: EnsembleBuilder
        An EnsembleBuilder created from the contents of a real autosklearn AutoML run

    Expects
    -------
    * Should generate cached items "ensemble_read_preds" and ensemble_read_losses"
    * Should generate an ensembles directory which contains at least one ensemble
    """
    builder.main(time_left=10, iteration=0)

    for path in [builder.run_predictions_path, builder.runs_path]:
        assert path.exists(), f"contents = {list(dir.iterdir())}"

    ens_dir = Path(builder.backend.get_ensemble_dir())

    assert ens_dir.exists()
    assert len(list(ens_dir.iterdir())) > 0
