import os
import shutil
import time
import unittest.mock

import dask
from dask.distributed import Client, get_client
import psutil
import pytest

from autosklearn.util.backend import create, Backend
from autosklearn.automl import AutoML


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
    output = os.path.join(
        test_dir, '.output__%s__%s' % (request.module.__name__, request.node.name)
    )

    for dir in (tmp, output):
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
        output,
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True,
    )

    def get_finalizer(tmp_dir, output_dir):
        def session_run_at_end():
            for dir in (tmp_dir, output_dir):
                for i in range(10):
                    if os.path.exists(dir):
                        try:
                            shutil.rmtree(dir)
                            break
                        except OSError:
                            time.sleep(1)
        return session_run_at_end
    request.addfinalizer(get_finalizer(tmp, output))

    return backend


@pytest.fixture(scope="function")
def tmp_dir(request):
    return _dir_fixture('tmp', request)


@pytest.fixture(scope="function")
def output_dir(request):
    return _dir_fixture('output', request)


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

    dask.config.set({'distributed.worker.daemon': False})
    client = Client(n_workers=2, threads_per_worker=1, processes=True)
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

    dask.config.set({'distributed.worker.daemon': False})
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
