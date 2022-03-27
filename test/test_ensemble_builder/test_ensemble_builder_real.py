from typing import Callable

from pathlib import Path
from shutil import rmtree

from autosklearn.automl import AutoML
from autosklearn.automl_common.common.utils.backend import Backend

from pytest_cases import parametrize_with_cases

import test.test_automl.cases as cases


@parametrize_with_cases("automl", cases=cases, has_tag="fitted")
def case_automl_cases(
    tmp_path: Path,
    automl: AutoML,
    make_backend: Callable[..., Backend],
) -> Backend:
    """Gives the backend for from the cached automl instance in `test_automl/cases.py`

    We do this by copying the backend produced from these cached automl runs to a new
    tmp directory for the ensemble builder tests to run from

    We also delete ensemble building specific things so that ensemble sees them as
    just a collection of runs and no previous ensemble building has been done.
    """
    original_backend = automl._backend
    backend_path = tmp_path / "backend"

    backend = make_backend(path=backend_path, template=original_backend)

    ensemble_dir = Path(backend.get_ensemble_dir())
    if ensemble_dir.exists():
        rmtree(ensemble_dir)

    ensemble_hist = Path(backend.internals_directory) / "ensemble_history.json"
    if ensemble_hist.exists():
        ensemble_hist.unlink()

    return backend


@parametrize_with_cases("ensemble_backend", cases=case_automl_cases)
def test_something(ensemble_backend: Backend) -> None:
    return
