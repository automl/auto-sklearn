import os
import shutil
import time
from typing import Tuple
import unittest.mock

from dask.distributed import Client, get_client
import numpy as np
import psutil
import pytest

from autosklearn.util.backend import create, Backend
import autosklearn.pipeline.util as putil
from autosklearn.automl import AutoML
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor


class AutoMLStub(AutoML):
    def __init__(self):
        self.__class__ = AutoML
        self._task = None
        self._dask_client = None
        self._is_dask_client_internally_created = False

    def __del__(self):
        pass


@pytest.fixture(scope="function")
def automl_stub(request):
    automl = AutoMLStub()
    automl._seed = 42
    automl._backend = unittest.mock.Mock(spec=Backend)
    automl._backend.context = unittest.mock.Mock()
    automl._delete_output_directories = lambda: 0
    return automl


@pytest.fixture(scope="function")
def backend(request):

    test_dir = os.path.dirname(__file__)
    tmp = os.path.join(test_dir, '.tmp__%s__%s' % (request.module.__name__, request.node.name))

    for dir in (tmp, ):
        for i in range(10):
            if os.path.exists(dir):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)

    # Make sure the folders we wanna create do not already exist.
    backend = create(
        tmp,
        delete_tmp_folder_after_terminate=True,
    )

    def get_finalizer(tmp_dir):
        def session_run_at_end():
            for dir in (tmp_dir, ):
                for i in range(10):
                    if os.path.exists(dir):
                        try:
                            shutil.rmtree(dir)
                            break
                        except OSError:
                            time.sleep(1)
        return session_run_at_end
    request.addfinalizer(get_finalizer(tmp))

    return backend


@pytest.fixture(scope="function")
def tmp_dir(request):
    return _dir_fixture('tmp', request)


@pytest.fixture(scope="module")
def tmp_dir_module_scope(request):
    return _dir_fixture('tmp', request)


def _dir_fixture(dir_type, request):

    test_dir = os.path.dirname(__file__)
    dir = os.path.join(
        test_dir, '.%s__%s__%s' % (dir_type, request.module.__name__, request.node.name)
    )

    for i in range(10):
        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                break
            except OSError:
                pass

    def get_finalizer(dir):
        def session_run_at_end():
            for i in range(10):
                if os.path.exists(dir):
                    try:
                        shutil.rmtree(dir)
                        break
                    except OSError:
                        time.sleep(1)

        return session_run_at_end

    request.addfinalizer(get_finalizer(dir))

    return dir


@pytest.fixture(scope="function")
def dask_client(request):
    """
    Create a dask client with two workers.

    Workers are in subprocesses to not create deadlocks with the pynisher and logging.
    """

    client = Client(n_workers=2, threads_per_worker=1, processes=False)
    print("Started Dask client={}\n".format(client))

    def get_finalizer(address):
        def session_run_at_end():
            client = get_client(address)
            print("Closed Dask client={}\n".format(client))
            client.shutdown()
            client.close()
            del client
        return session_run_at_end
    request.addfinalizer(get_finalizer(client.scheduler_info()['address']))

    return client


@pytest.fixture(scope="function")
def dask_client_single_worker(request):
    """
    Same as above, but only with a single worker.

    Using this might cause deadlocks with the pynisher and the logging module. However,
    it is used very rarely to avoid this issue as much as possible.
    """

    client = Client(n_workers=1, threads_per_worker=1, processes=False)
    print("Started Dask client={}\n".format(client))

    def get_finalizer(address):
        def session_run_at_end():
            client = get_client(address)
            print("Closed Dask client={}\n".format(client))
            client.shutdown()
            client.close()
            del client
        return session_run_at_end
    request.addfinalizer(get_finalizer(client.scheduler_info()['address']))

    return client


def pytest_sessionfinish(session, exitstatus):
    proc = psutil.Process()
    for child in proc.children(recursive=True):
        print(child, child.cmdline())


@pytest.fixture(scope='session')
def iris_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Provides the 'iris' multi-label classification dataset """
    return putil.get_dataset('iris')  # type: ignore


@pytest.fixture(scope='session')
def boston_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Provides the 'boston' multi-label classification dataset """
    return putil.get_dataset('boston')  # type: ignore


@pytest.fixture(scope='module')
def simple_AutoSklearnClassifier(
    tmp_dir_module_scope: str,
    iris_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> AutoSklearnClassifier:
    """ Provides a standard fitted re-usable AutoSklearnClassifier for tests.

    Useful for read-only testing, not for any functions which might modify state.

    Creates a classifier as a user might make to test auto-sklearn
    on the iris dataset, mostly keeping default parameters.
    Keeps as close to the first example seen at
        https://automl.github.io/auto-sklearn/master/index.html#auto-sklearn

    Tests which rely on the following should make their own instances:
        * Specific construction parameters
        * Specific fitting parameters or data
        * Performance (by metric or time)
        * Calling any state modifying functions
        * Manually modifying state
    """
    X_train, Y_train, _, _ = iris_dataset
    classifier = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=tmp_dir_module_scope,
        seed=1
    )
    classifier.fit(X_train, Y_train)
    return classifier


@pytest.fixture(scope='module')
def simple_AutoSklearnRegressor(
    tmp_dir_module_scope: str,
    boston_dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> AutoSklearnRegressor:
    """ Provides a standard fitted re-usable AutoSklearnRegressor for tests.

    Useful for read-only testing, not for any functions which might modify state.

    Creates a classifier as a user might make to test auto-sklearn
    on the iris dataset, mostly keeping default parameters.
    Keeps as close to the first example seen at
        https://automl.github.io/auto-sklearn/master/index.html#auto-sklearn

    Tests which rely on the following should make their own instances:
        * Specific construction parameters
        * Specific fitting parameters or data
        * Performance (by metric or time)
        * Calling any state modifying functions
        * Manually modifying state
    """
    X_train, Y_train, _, _ = boston_dataset
    regressor = AutoSklearnRegressor(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        tmp_folder=tmp_dir_module_scope,
        seed=1
    )
    regressor.fit(X_train, Y_train)
    return regressor
