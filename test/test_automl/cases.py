"""
Here we define the several different setups and cache them to allow for easier unit
testing. The caching mechanism is only per session and does not persist over sessions.
There's only really a point for caching fitted models that will be tested multiple times
for different properties.

Anything using a cached model must not destroy any backend resources, although it can
mock if required.

Tags:
    {classifier, regressor} - The type of AutoML object
        classifier - will be fit on "iris"
        regressor - will be fit on "boston"
    {fitted} - If the automl case has been fitted
    {cv, holdout} - Whether explicitly cv or holdout was used
    {no_ensemble} - Fit with no ensemble size
    {multiobjective} - If the automl instance is multiobjective
"""
from __future__ import annotations

from typing import Callable, Tuple

from pathlib import Path

import numpy as np
import sklearn.model_selection

import autosklearn.metrics
from autosklearn.automl import AutoMLClassifier, AutoMLRegressor
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.evaluation.abstract_evaluator import (
    MyDummyClassifier,
    MyDummyRegressor,
)

from pytest_cases import case, parametrize

from test.fixtures.backend import copy_backend
from test.fixtures.caching import Cache


def stop_at_first(smbo, run_info, result, time_left) -> bool:
    """Used in some cases to enforce the only valid model is the dummy model"""
    return False


@case(tags=["classifier"])
def case_classifier(
    tmp_dir: str,
    make_automl_classifier: Callable[..., AutoMLClassifier],
) -> AutoMLClassifier:
    """Case basic unfitted AutoMLClassifier"""
    dir = Path(tmp_dir) / "backend"
    model = make_automl_classifier(temporary_directory=str(dir))
    return model


@case(tags=["classifier"])
def case_regressor(
    tmp_dir: str,
    make_automl_regressor: Callable[..., AutoMLRegressor],
) -> AutoMLRegressor:
    """Case basic unfitted AutoMLClassifier"""
    dir = Path(tmp_dir) / "backend"
    model = make_automl_regressor(temporary_directory=str(dir))
    return model


# ###################################
# The following are fitted and cached
# ###################################
@case(tags=["classifier", "fitted", "holdout"])
@parametrize("dataset", ["iris"])
def case_classifier_fitted_holdout_iterative(
    dataset: str,
    make_cache: Callable[[str], Cache],
    make_backend: Callable[..., Backend],
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a holdout fitted classifier"""
    resampling_strategy = "holdout-iterative-fit"

    key = f"case_classifier_{resampling_strategy}_{dataset}"

    # This locks the cache for this item while we check, required for pytest-xdist
    with make_cache(key) as cache:
        if "model" not in cache:
            # Make the model in the cache
            model = make_automl_classifier(
                temporary_directory=cache.path("backend"),
                delete_tmp_folder_after_terminate=False,
                resampling_strategy=resampling_strategy,
            )

            X, y, Xt, yt = make_sklearn_dataset(name=dataset)
            model.fit(X, y, dataset_name=dataset)

            # Save the model
            cache.save(model, "model")

    # Try the model from the cache
    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["classifier", "fitted", "cv"])
@parametrize("dataset", ["iris"])
def case_classifier_fitted_cv(
    make_cache: Callable[[str], Cache],
    dataset: str,
    make_backend: Callable[..., Backend],
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a fitted cv AutoMLClassifier"""
    resampling_strategy = "cv"

    key = f"case_classifier_{resampling_strategy}_{dataset}"

    # This locks the cache for this item while we check, required for pytest-xdist
    with make_cache(key) as cache:
        if "model" not in cache:
            model = make_automl_classifier(
                resampling_strategy=resampling_strategy,
                temporary_directory=cache.path("backend"),
                delete_tmp_folder_after_terminate=False,
                time_left_for_this_task=60,  # Give some more for CV
                per_run_time_limit=10,
            )

            X, y, Xt, yt = make_sklearn_dataset(name=dataset)
            model.fit(X, y, dataset_name=dataset)

            cache.save(model, "model")

    # Try the model from the cache
    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["classifier", "fitted", "holdout", "multiobjective"])
@parametrize("dataset", ["iris"])
def case_classifier_fitted_holdout_multiobjective(
    dataset: str,
    make_cache: Callable[[str], Cache],
    make_backend: Callable[..., Backend],
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a holdout fitted classifier"""
    resampling_strategy = "holdout"

    key = f"case_classifier_{resampling_strategy}_{dataset}"

    # This locks the cache for this item while we check, required for pytest-xdist
    with make_cache(key) as cache:
        if "model" not in cache:
            # Make the model in the cache
            model = make_automl_classifier(
                temporary_directory=cache.path("backend"),
                delete_tmp_folder_after_terminate=False,
                resampling_strategy=resampling_strategy,
                metrics=[
                    autosklearn.metrics.balanced_accuracy,
                    autosklearn.metrics.log_loss,
                ],
            )

            X, y, Xt, yt = make_sklearn_dataset(
                name=dataset, return_target_as_string=True
            )
            model.fit(X, y, dataset_name=dataset)

            # Save the model
            cache.save(model, "model")

    # Try the model from the cache
    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["regressor", "fitted", "holdout"])
@parametrize("dataset", ["boston"])
def case_regressor_fitted_holdout(
    make_cache: Callable[[str], Cache],
    dataset: str,
    make_backend: Callable[..., Backend],
    make_automl_regressor: Callable[..., AutoMLRegressor],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLRegressor:
    """Case of fitted regressor with holdout"""
    resampling_strategy = "holdout"

    key = f"case_regressor_{resampling_strategy}_{dataset}"

    # This locks the cache for this item while we check, required for pytest-xdist
    with make_cache(key) as cache:
        if "model" not in cache:
            model = make_automl_regressor(
                temporary_directory=cache.path("backend"),
                resampling_strategy=resampling_strategy,
                delete_tmp_folder_after_terminate=False,
            )

            X, y, Xt, yt = make_sklearn_dataset(name=dataset)
            model.fit(X, y, dataset_name=dataset)

            cache.save(model, "model")

    # Try the model from the cache
    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["regressor", "fitted", "cv"])
@parametrize("dataset", ["boston"])
def case_regressor_fitted_cv(
    make_cache: Callable[[str], Cache],
    dataset: str,
    make_backend: Callable[..., Backend],
    make_automl_regressor: Callable[..., AutoMLRegressor],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLRegressor:
    """Case of fitted regressor with cv resampling"""
    resampling_strategy = "cv"
    key = f"case_regressor_{resampling_strategy}_{dataset}"

    # This locks the cache for this item while we check, required for pytest-xdist
    with make_cache(key) as cache:

        if "model" not in cache:
            model = make_automl_regressor(
                temporary_directory=cache.path("backend"),
                resampling_strategy=resampling_strategy,
                delete_tmp_folder_after_terminate=False,
                time_left_for_this_task=60,  # Some extra time for CV
                per_run_time_limit=10,
            )

            X, y, Xt, yt = make_sklearn_dataset(name=dataset)
            model.fit(X, y, dataset_name=dataset)

            cache.save(model, "model")

    # Try the model from the cache
    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["classifier", "fitted", "no_ensemble"])
@parametrize("dataset", ["iris"])
def case_classifier_fitted_no_ensemble(
    make_cache: Callable[[str], Cache],
    dataset: str,
    make_backend: Callable[..., Backend],
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a fitted classifier but ensemble was disabled"""
    key = f"case_classifier_fitted_no_ensemble_{dataset}"

    # This locks the cache for this item while we check, required for pytest-xdist
    with make_cache(key) as cache:

        if "model" not in cache:
            model = make_automl_classifier(
                temporary_directory=cache.path("backend"),
                delete_tmp_folder_after_terminate=False,
                ensemble_class=None,
            )

            X, y, Xt, yt = make_sklearn_dataset(name=dataset)
            model.fit(X, y, dataset_name=dataset)

            cache.save(model, "model")

    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["classifier", "fitted"])
def case_classifier_fitted_only_dummy(
    make_cache: Callable[[str], Cache],
    make_backend: Callable[..., Backend],
    make_automl_classifier: Callable[..., AutoMLClassifier],
) -> AutoMLClassifier:
    """Case of a fitted classifier but only dummy was found"""
    key = "case_classifier_fitted_only_dummy"

    # This locks the cache for this item while we check, required for pytest-xdist

    with make_cache(key) as cache:
        if "model" not in cache:
            model = make_automl_classifier(
                temporary_directory=cache.path("backend"),
                delete_tmp_folder_after_terminate=False,
                include={"classifier": ["bernoulli_nb"]},  # Just a meh model
                get_trials_callback=stop_at_first,
            )
            rand = np.random.RandomState(2)
            _X = rand.random((100, 50))
            _y = rand.randint(0, 2, (100,))
            X, Xt, y, yt = sklearn.model_selection.train_test_split(
                _X, _y, random_state=1  # Required to ensure dummy is best
            )
            model.fit(X, y, dataset_name="random")

            # We now validate that indeed, the only model is the Dummy
            members = list(model.models_.values())
            if len(members) != 1 and not isinstance(members[0], MyDummyClassifier):
                raise ValueError("Should only have one model, dummy\n", members)

            cache.save(model, "model")

    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model


@case(tags=["regressor", "fitted"])
def case_regressor_fitted_only_dummy(
    make_cache: Callable[[str], Cache],
    make_backend: Callable[..., Backend],
    make_automl_regressor: Callable[..., AutoMLRegressor],
) -> AutoMLRegressor:
    """Case of a fitted classifier but only dummy was found"""
    key = "case_regressor_fitted_only_dummy"

    # This locks the cache for this item while we check, required for pytest-xdist

    with make_cache(key) as cache:
        if "model" not in cache:
            model = make_automl_regressor(
                temporary_directory=cache.path("backend"),
                delete_tmp_folder_after_terminate=False,
                include={"regressor": ["k_nearest_neighbors"]},  # Just a meh model
                get_trials_callback=stop_at_first,
            )

            rand = np.random.RandomState(2)
            _X = rand.random((100, 50))
            _y = rand.random((100,))

            X, Xt, y, yt = sklearn.model_selection.train_test_split(
                _X, _y, random_state=1  # Required to ensure dummy is best
            )
            model.fit(X, y, dataset_name="random")

            # We now validate that indeed, the only model is the Dummy
            members = list(model.models_.values())
            if len(members) != 1 and not isinstance(members[0], MyDummyRegressor):
                raise ValueError("Should only have one model, dummy\n", members)

            cache.save(model, "model")

    model = cache.load("model")
    model._backend = copy_backend(old=model._backend, new=make_backend())

    return model
