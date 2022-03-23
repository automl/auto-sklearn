from __future__ import annotations

from typing import Any, Callable, Optional

import pickle
import shutil
import traceback
from functools import partial
from pathlib import Path

from autosklearn.automl import AutoML

from pytest import FixtureRequest
from pytest_cases import fixture


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
        self.dir = cache_dir / key
        self.verbose = verbose > 0

    def items(self) -> list[Path]:
        """Get any paths associated to items in this dir"""
        return list(self.dir.iterdir())

    def __contains__(self, name: str) -> bool:
        return self.path(name).exists()

    def path(self, name: str) -> Path:
        """Path to an item for this cache"""
        return self.dir / name

    def _load(self, name: str) -> Any:
        """Load an item from the cache with a given name"""
        if self.verbose:
            print(f"Loading cached item {self.path(name)}")

        with self.path(name).open("rb") as f:
            return pickle.load(f)

    def _save(self, item: Any, name: str) -> None:
        """Dump an item to cache with a name"""
        if self.verbose:
            print(f"Saving cached item {self.path(name)}")

        with self.path(name).open("wb") as f:
            pickle.dump(item, f)

    def reset(self) -> None:
        """Delete this caches items"""
        shutil.rmtree(self.dir)
        self.dir.mkdir()


class AutoMLCache(Cache):
    def save(self, model: AutoML) -> None:
        """Save the model"""
        self._save(model, "model")

    def model(self) -> Optional[AutoML]:
        """Returns the saved model if it can.

        In the case of an issue loading an existing model file, it will delete
        this cache item.
        """
        if "model" not in self:
            return None

        # Try to load the model, if there was an issue, delete all cached items
        # for the model and return None
        try:
            model = self._load("model")
        except Exception:
            model = None
            print(traceback.format_exc())
            self.reset()
        finally:
            return model

    def backend_path(self) -> Path:
        """The path for the backend of the automl model"""
        return self.path("backend")


@fixture
def cache(request: FixtureRequest) -> Callable[[str], Cache]:
    """Gives the access to a cache."""
    pytest_cache = request.config.cache
    assert pytest_cache is not None

    cache_dir = pytest_cache.mkdir("autosklearn-cache")
    return partial(Cache, cache_dir=cache_dir)


@fixture
def automl_cache(request: FixtureRequest) -> Callable[[str], AutoMLCache]:
    """Gives access to an automl cache"""
    pytest_cache = request.config.cache
    assert pytest_cache is not None

    cache_dir = pytest_cache.mkdir("autosklearn-cache")
    verbosity = request.config.getoption("verbose")
    return partial(AutoMLCache, cache_dir=cache_dir, verbose=verbosity)
