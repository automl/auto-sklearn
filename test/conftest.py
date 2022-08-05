"""
Testing
=======
The following are some features, guidelines and functionality for testing which makes
updating, adding and refactoring tests easier, especially as features and functionality
changes.

**Marks**
* todo - ``pytest.mark.todo``` to mark a test which xfails as it's todo
* slow - ``pytest.mark.slow``` to mark a test which is skipped if `pytest --fast`

**Documenting Tests**
To ease in understanding of tests, what is being tested and what's expected of the test,
each test should doc expected bhaviour, not describe the steps of the test.
Commenst relating to how a test does things can be left in the tests and not the doc.

    Expects
    -------
    * Something should raise a ValueError when called with X as X is not handled by the
      validator Y.

**Test same module across files**
When a module has many complicated avenues to be tested, create a folder and split the
tests according to each avenue. See `test/test_automl` for example as the `automl.py`
module is quite complicated to test and all tests in a single file become difficult to
follow and change.

**pytest_cases**
Using pytest_cases, we seperate a `case`, something that defines the state of the
object, from the actual `test`, which tests properties of these cases.

A complicated example can be seen at `test/test_automl/cases.py` where we have
autoML instances that are classifier/regressor, fitted or not, with cv or holdout,
or fitted with no ensemble. TODO: Easier example.

Docs: https://smarie.github.io/python-pytest-cases/

**Caching**
Uses pytest's cache functionality for long training models so they can be shared between
tests and between different test runs. This is primarly used with `cases` so that tests
requiring the same kind of expensive case and used cached values.

Use `pytest --cached` to use this feature.

See `test/test_automl/cases.py` for example of how the fixtures from
`test/fixtures/caching.py` can be used to cache objects between tests.

**Fixtures**
All fixtures in "test/fixtures" are known in every test file. We try to make use
of fixture `factories` which can be used to construct objects in complicated ways,
removing these complications from the tests themselves, importantly, keeping tests
short. A convention we use is to prefix them with `make`, for example,
`make_something`. This is useful for making data, e.g. `test/fixtures/data::make_data`

..code:: python

    # Example of fixture factory
    @fixture
    def make_something():
        def _make(...args):
            # ... complicated setup
            # ... more complications
            # ... make some sub objects which are complicated
            return something

        return _make

    @parametrize("arg1", ['a', 'b', 'c'])
    def test_something_does_x(arg1, make_something):
        something = make_something(arg1, ...)
        result = something.run()
        assert something == expected
"""
from typing import Any, Iterator, List, Optional

import re
import shutil
import signal
from pathlib import Path

import psutil

import pytest
from pytest import ExitCode, Item, Session

DEFAULT_SEED = 0


HERE = Path(__file__)
AUTOSKLEARN_CACHE_NAME = "autosklearn-cache"


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
    todos = [marker for marker in item.iter_markers(name="todo")]
    if todos:
        pytest.xfail(f"Test needs to be implemented, {item.location}")


def pytest_sessionstart(session: Session) -> None:
    """Called after the ``Session`` object has been created and before performing
    collection and entering the run test loop.

    Parameters
    ----------
    session : Session
        The pytest session object
    """
    config = session.config
    cache = config.cache

    if cache is None:
        return

    # We specifically only remove the cached items dir, not any information
    # about previous tests which also exist in `.pytest_cache`
    if not config.getoption("--cached"):
        dir = cache.mkdir(AUTOSKLEARN_CACHE_NAME)
        shutil.rmtree(dir)

    return


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    """Clean up any child processes"""
    proc = psutil.Process()
    kill_signal = signal.SIGTERM
    for child in proc.children(recursive=True):

        # https://stackoverflow.com/questions/57336095/access-verbosity-level-in-a-pytest-helper-function
        if session.config.getoption("verbose") > 0:
            print(child, child.cmdline())

        # https://psutil.readthedocs.io/en/latest/#kill-process-tree
        try:
            child.send_signal(kill_signal)
        except psutil.NoSuchProcess:
            pass


Config = Any  # Can't find import?


def pytest_collection_modifyitems(
    session: Session,
    config: Config,
    items: List[Item],
) -> None:
    """Modifys the colelction of tests that are captured"""
    if config.getoption("--fast"):
        skip = pytest.mark.skip(reason="Test marked `slow` and `--fast` arg used")

        slow_items = [item for item in items if "slow" in item.keywords]
        for item in slow_items:
            item.add_marker(skip)


def pytest_configure(config: Config) -> None:
    """Used to register marks"""
    config.addinivalue_line("markers", "todo: Mark test as todo")


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
        "--fast",
        action="store_true",
        default=False,
        help="Disable tests marked as slow",
    )
    parser.addoption(
        "--cached",
        action="store_true",
        default=False,
        help="Cache everything between invocations of pytest",
    )
