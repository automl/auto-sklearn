from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Type

from functools import partial

from autosklearn.automl import AutoML, AutoMLClassifier, AutoMLRegressor
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.ensembles.ensemble_selection import EnsembleSelection

from pytest import FixtureRequest, fixture
from unittest.mock import Mock

from test.conftest import DEFAULT_SEED
from test.fixtures.dask import create_test_dask_client


def _create_automl(
    automl_type: Type[AutoML] = AutoML,
    _id: str | None = None,
    **kwargs: Any,
) -> AutoML:
    """

    Parameters
    ----------
    automl_type : Type[AutoML] = AutoML
        The type of AutoML object to use

    _id: str | None = None
        If no dask client is provided, a unique id is required to create one
        so that it can be shut down after the test ends

    **kwargs: Any
        Options to pass on to the AutoML type for construction

    Returns
    -------
    AutoML
        The constructed class and a close method for dask, if it exists
    """
    test_defaults = {
        "n_jobs": 2,
        "time_left_for_this_task": 30,
        "per_run_time_limit": 5,
        "seed": DEFAULT_SEED,
        "ensemble_class": EnsembleSelection,
        "ensemble_kwargs": {"ensemble_size": 10},
        "ensemble_nbest": 10,
        "max_models_on_disc": 10,
        "initial_configurations_via_metalearning": 5,
    }

    # If a temp directory was explicitly passed, don't delete it automatically
    # Normally the `tmp_path` fixutre will delete it anyways
    if "temporary_directory" in kwargs:
        test_defaults["delete_tmp_folder_after_terminate"] = False

    opts: Dict[str, Any] = {**test_defaults, **kwargs}

    if "dask_client" not in opts:
        assert _id is not None
        client = create_test_dask_client(id=_id, n_workers=opts["n_jobs"])
        opts["dask_client"] = client

    auto = automl_type(**opts)
    return auto


@fixture
def make_automl(request: FixtureRequest) -> Callable[..., Tuple[AutoML, Callable]]:
    """See `_create_automl`"""
    yield partial(_create_automl, automl_type=AutoML, _id=request.node.nodeid)


@fixture
def make_automl_classifier(request: FixtureRequest) -> Callable[..., AutoMLClassifier]:
    """See `_create_automl`"""
    yield partial(_create_automl, automl_type=AutoMLClassifier, _id=request.node.nodeid)


@fixture
def make_automl_regressor(request: FixtureRequest) -> Callable[..., AutoMLRegressor]:
    """See `_create_automl`"""
    yield partial(_create_automl, automl_type=AutoMLRegressor, _id=request.node.nodeid)


class AutoMLStub(AutoML):
    def __init__(self) -> None:
        self.__class__ = AutoML
        self._task = None
        self._dask_client = None  # type: ignore
        self._is_dask_client_internally_created = False

    def __del__(self) -> None:
        pass


@fixture(scope="function")
def automl_stub() -> AutoMLStub:
    """TODO remove"""
    automl = AutoMLStub()
    automl._seed = 42
    automl._backend = Mock(spec=Backend)
    automl._backend.context = Mock()
    automl._delete_output_directories = lambda: 0
    return automl
