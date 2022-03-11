from typing
import os
import shutil
import time
import unittest.mock

import psutil
import pytest
from dask.distributed import Client, get_client

from autosklearn.automl import AutoML
from autosklearn.automl_common.common.utils.backend import Backend, create


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
    tmp = os.path.join(
        test_dir, ".tmp__%s__%s" % (request.module.__name__, request.node.name)
    )

    for dir in (tmp,):
        for i in range(10):
            if os.path.exists(dir):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)

    # Make sure the folders we wanna create do not already exist.
    backend = create(
        temporary_directory=tmp, output_directory=None, prefix="auto-sklearn"
    )

    def get_finalizer(tmp_dir):
        def session_run_at_end():
            for dir in (tmp_dir,):
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
    return _dir_fixture("tmp", request)


def _dir_fixture(dir_type, request):
    test_dir = os.path.dirname(__file__)

    dirname = f".{dir_type}__{request.module.__name__}__{request.node.name}"
    dir = os.path.join(test_dir, dirname)

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

    request.addfinalizer(get_finalizer(client.scheduler_info()["address"]))

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

    request.addfinalizer(get_finalizer(client.scheduler_info()["address"]))

    return client


def pytest_sessionfinish(session, exitstatus):
    proc = psutil.Process()
    for child in proc.children(recursive=True):
        print(child, child.cmdline())


def walk(path: Path, include: Optional[str] = None) -> Iterator[Path]:
    """Yeilds all files, iterating over directory

    Parameters
    ----------
    path: Path
        The root path to walk from

    include: Optional[str] = None
        Include only directories which match this string

    Returns
    -------
    Iterator[Path]
        All file paths that could be found from this walk
    """
    for p in path.iterdir():
        if p.is_dir():
            if include is None or re.match(include, p.name):
                yield from walk(p, include)
        else:
            yield p.resolve()


def is_fixture(path: Path) -> bool:
    """Whether a path is a fixture"""
    return path.name.endswith("fixtures.py")


def as_module(path: Path) -> str:
    """Convert a path to a module as seen from here"""
    root = here.parent.parent
    parts = path.relative_to(root).parts
    return ".".join(parts).replace(".py", "")


def fixture_modules() -> List[str]:
    """Get all fixture modules"""
    fixtures_folder = here.parent / "fixtures"
    return [
        as_module(path) for path in walk(fixtures_folder) if path.name.endswith(".py")
    ]

pytest_plugins += fixture_modules()

def pytest_runtest_setup(item: Item) -> None:
    """Run before each test"""
    todos = [mark for mark in item.iter_markers(name="todo")]
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")
