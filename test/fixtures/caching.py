from __future__ import annotations

from typing import Any, Callable

import os
import pickle
import shutil
import tempfile
from functools import partial
from pathlib import Path

from filelock import FileLock

from pytest import FixtureRequest
from pytest_cases import fixture

from test.conftest import AUTOSKLEARN_CACHE_NAME

LOCK_DIR = Path(tempfile.gettempdir())


class Cache:
    """Used for the below fixtures.

    Mainly used with cases so they don' need to be retrained at every invocation.
    The cache can be cleared using `pytest`'s built in mechanism:

        pytest --clear-cache

    To view cached items use:

        pytest --cache-show

    ..code:: python

        def case_fitted_model(cache, ...):
            key = "some key unique to this test"
            cache = cache(key)
            if "model" in cache:
                return cache.load("model")

            # ... fit model

            cache.save(model, "model")
            return model

    If multiple items are required, they can be access in much the same ways

    ..code:: python

        def case_requires_multiple_things(cache, ...):

            cache1 = cache("key1")
            cache2 = cache("key2")

    If multiple things need to be stored in one location, you can access a given path
    for a given named thing inside a cache.

    ..code:: python

        def case_fitted_model_and_populated_backend(cache, ...):
            cache = cache("some key")

    """

    def __init__(self, key: str, cache_dir: Path, verbose: int = 0):
        """
        Parameters
        ----------
        key : str
            The key of the item stored

        cache_dir : Path
            The dir where the cache should be located

        verbose : int = 0
            Whether to be verbose or not. Currently only has one level (> 0)
        """
        self.key = key
        self.dir = cache_dir / key
        self.verbose = verbose > 0
        self._lock: FileLock = None

    def items(self) -> list[Path]:
        """Get any paths associated to items in this dir"""
        return list(self.dir.iterdir())

    def __contains__(self, name: str) -> bool:
        return self.path(name).exists()

    def path(self, name: str) -> Path:
        """Path to an item for this cache"""
        return self.dir / name

    def load(self, name: str) -> Any:
        """Load an item from the cache with a given name"""
        if self.verbose:
            print(f"Loading cached item {self.path(name)}")

        with self.path(name).open("rb") as f:
            return pickle.load(f)

    def save(self, item: Any, name: str) -> None:
        """Dump an item to cache with a name"""
        if self.verbose:
            print(f"Saving cached item {self.path(name)}")

        with self.path(name).open("wb") as f:
            pickle.dump(item, f)

    def reset(self) -> None:
        """Delete this caches items"""
        shutil.rmtree(self.dir)
        self.dir.mkdir()

    def __enter__(self):
        if int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 1)) <= 1:
            return self
        else:
            path = LOCK_DIR / f"{self.key}.lock"
            self._lock = FileLock(path)
            self._lock.acquire(poll_interval=1.0)
            if self.verbose:
                print(f"locked cache {path}")

            return self

    def __exit__(self, *args, **kwargs):
        if self._lock is not None:
            self._lock.release()


@fixture
def make_cache(request: FixtureRequest) -> Callable[[str], Cache]:
    """Gives the access to a cache."""
    pytest_cache = request.config.cache
    assert pytest_cache is not None

    cache_dir = pytest_cache.mkdir(AUTOSKLEARN_CACHE_NAME)
    return partial(Cache, cache_dir=cache_dir)
