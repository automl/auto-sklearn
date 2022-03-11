"""
Testing
=======

**Features**
* marker - ``pytest.mark.todo``` to mark a test which xfails as it's todo
* fixtures - All fixtures in "test/fixtures" are known in every test file
"""
from typing import Any, Iterator, List, Optional

import re
from pathlib import Path

import psutil
import pytest
from pytest import ExitCode, Item, Session

HERE = Path(__file__)


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


def pytest_runtest_setup(item: Item) -> None:
    """Run before each test"""
    todos = [mark for mark in item.iter_markers(name="todo")]
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    """Clean up any chil processes"""
    proc = psutil.Process()
    for child in proc.children(recursive=True):
        print(child, child.cmdline())


Config = Any  # Can't find import?


def pytest_collection_modifyitems(
    session: Session,
    config: Config,
    items: List[Item],
) -> None:
    """Modifys the colelction of tests that are captured"""
    if config.getoption("--fast-only"):
        skip_slow = pytest.mark.skip(reason="Test is marked as slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


pytest_plugins = fixture_modules()


Parser = Any  # Can't find import?


def pytest_addoption(parser: Parser) -> None:
    """

    Parameters
    ----------
    parser : Parser
        The parser to add options to
    """
    parser.addoption(
        "--fast-only",
        action="store_true",
        default=False,
        help="Disable tests marked as slow",
    )
