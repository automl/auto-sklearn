from unittest.mock import Mock

from pytest import fixture

from autosklearn.automl import AutoML
from autosklearn.automl_common.common.utils.backend import Backend


class AutoMLStub(AutoML):
    def __init__(self) -> None:
        self.__class__ = AutoML
        self._task = None
        self._dask_client = None
        self._is_dask_client_internally_created = False

    def __del__(self) -> None:
        pass


@fixture(scope="function")
def automl_stub() -> AutoMLStub:
    automl = AutoMLStub()
    automl._seed = 42
    automl._backend = Mock(spec=Backend)
    automl._backend.context = Mock()
    automl._delete_output_directories = lambda: 0
    return automl
