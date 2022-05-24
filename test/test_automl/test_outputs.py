from pathlib import Path

from autosklearn.automl import AutoML
from autosklearn.ensemble_building.builder import CANDIDATES_FILENAME

from pytest import mark
from pytest_cases import parametrize_with_cases
from pytest_cases.filters import has_tag

import test.test_automl.cases as cases
from test.conftest import DEFAULT_SEED

# Some filters
has_ensemble = has_tag("fitted") & ~has_tag("no_ensemble")
no_ensemble = has_tag("fitted") & has_tag("no_ensemble")


@mark.todo
def test_datamanager_stored_contents() -> None:
    ...


@parametrize_with_cases("automl", cases=cases, filter=has_ensemble)
def test_paths_created(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        A previously fitted automl

    Expects
    -------
    * The given paths should exist after the automl has been run and fitted
    """
    assert automl._backend is not None

    partial = Path(automl._backend.internals_directory)
    expected = [
        partial / fixture
        for fixture in (
            "true_targets_ensemble.npy",
            f"start_time_{DEFAULT_SEED}",
            "datamanager.pkl",
            "runs",
        )
    ]

    for path in expected:
        assert path.exists()


@parametrize_with_cases("automl", cases=cases, filter=has_ensemble)
def test_paths_created_with_ensemble(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        A previously fitted automl

    Expects
    -------
    * The given paths for an automl with an ensemble should include paths
    specific to ensemble building
    """
    assert automl._backend is not None

    partial = Path(automl._backend.internals_directory)
    expected = [
        partial / fixture
        for fixture in (
            "ensembles",
            "ensemble_history.json",
            CANDIDATES_FILENAME,
        )
    ]

    for path in expected:
        assert path.exists()


@parametrize_with_cases("automl", cases=cases, filter=has_ensemble)
def test_at_least_one_model_and_predictions(automl: AutoML) -> None:
    """
    Expects
    -------
    * There should be at least one models saved
    * Each model saved should have predictions for the ensemble
    """
    assert automl._backend is not None
    runs_dir = Path(automl._backend.get_runs_directory())

    runs = list(runs_dir.iterdir())
    assert len(runs) > 0

    at_least_one = False
    for run in runs:
        prediction_files = run.glob("predictions_ensemble*.npy")
        model_files = run.glob("*.*.model")

        if any(prediction_files):
            at_least_one = True
            assert any(model_files), "Run produced prediction but no model"

    assert at_least_one, "No runs produced predictions"


@parametrize_with_cases("automl", cases=cases, filter=has_ensemble)
def test_at_least_one_ensemble(automl: AutoML) -> None:
    """
    Expects
    -------
    * There should be at least one ensemble generated
    """
    assert automl._backend is not None
    ens_dir = Path(automl._backend.get_ensemble_dir())

    # TODO make more generic
    assert len(list(ens_dir.glob("*.ensemble"))) > 0
