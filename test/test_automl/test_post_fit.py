"""Check the internal state of the automl instances after it has been fitted"""

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


@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "holdout"])
def test_holdout_loaded_models(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        The fitted automl object to test

    Expects
    -------
    * The ensemble should not be empty
    * The models_ should contain the identifiers of what's in the ensemble
    * The cv_models_ attr should remain None
    """
    assert automl.ensemble_ is not None

    ensemble_identifiers = automl.ensemble_.get_selected_model_identifiers()

    assert set(automl.models_.keys()) == set(ensemble_identifiers)
    assert automl.cv_models_ is None


@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "cv"])
def test_cv_loaded_models(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        The fitted automl object to test

    Expects
    -------
    * The ensemble should not be empty
    * The models_ should contain the identifiers of what's in the ensemble
    * The cv_models_ should contain the identifiers of what's in the ensemble
    """
    assert automl.ensemble_ is not None

    ensemble_identifiers = automl.ensemble_.get_selected_model_identifiers()

    assert set(automl.models_.keys()) == set(ensemble_identifiers)
    assert set(automl.cv_models_.keys()) == set(ensemble_identifiers)


@parametrize_with_cases("automl", cases=cases, has_tag=no_ensemble)
def test_no_ensemble(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        A fitted automl object with ensemble size specified as 0

    Expects
    -------
    * Auto-sklearn does not load a model
    * The models_ should be of size 0
    * The cv_models_ should remain None
    """
    assert automl.ensemble_ is None
    assert len(automl.models_) == 0
    assert len(automl.cv_models_) == 0


@mark.todo
def test_datamanager_stored_contents() -> None:
    """
    Expects
    -------
    * TODO
    """
    ...


@parametrize_with_cases("automl", cases=cases, filter=has_ensemble)
def test_paths_created(automl: AutoML) -> None:
    """
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
