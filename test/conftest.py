from typing import Any, Iterator, List, Optional, Callable

import os
import re
import shutil
import time
import unittest.mock
from pathlib import Path

import psutil
import pytest
from dask.distributed import Client, get_client
from pytest import ExitCode, FixtureRequest, Item, Session

from autosklearn.automl import AutoML
from autosklearn.automl_common.common.utils.backend import Backend, create

HERE = Path(__file__)


class AutoMLStub(AutoML):
    def __init__(self) -> None:
        self.__class__ = AutoML
        self._task = None
        self._dask_client = None
        self._is_dask_client_internally_created = False

    def __del__(self) -> None:
        pass


@pytest.fixture(scope="function")
def automl_stub(request: FixtureRequest) -> AutoMLStub:
    automl = AutoMLStub()
    automl._seed = 42
    automl._backend = unittest.mock.Mock(spec=Backend)
    automl._backend.context = unittest.mock.Mock()
    automl._delete_output_directories = lambda: 0
    return automl


@pytest.fixture(scope="function")
def backend(request: FixtureRequest) -> Backend:
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

    def get_finalizer(tmp_dir: str) -> Callable:
        def session_run_at_end() -> None:
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
def tmp_dir(request: FixtureRequest) -> str:
    return _dir_fixture("tmp", request)


def _dir_fixture(dir_type: str, request: FixtureRequest) -> str:
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

    def get_finalizer(dir: str) -> Callable:
        def session_run_at_end() -> None:
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
    root = HERE.parent.parent
    parts = path.relative_to(root).parts
    return ".".join(parts).replace(".py", "")


def fixture_modules() -> List[str]:
    """Get all fixture modules"""
    fixtures_folder = HERE.parent / "fixtures"
    return [
        as_module(path) for path in walk(fixtures_folder) if path.name.endswith(".py")
    ]


pytest_plugins = fixture_modules()


def pytest_runtest_setup(item: Item) -> None:
    """Run before each test"""
    todos = [mark for mark in item.iter_markers(name="todo")]
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    proc = psutil.Process()
    for child in proc.children(recursive=True):
        print(child, child.cmdline())
