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
"""
from typing import Callable, Tuple

from pathlib import Path

import numpy as np

from autosklearn.automl import AutoMLClassifier, AutoMLRegressor

from pytest_cases import case, parametrize

from test.fixtures.caching import AutoMLCache


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
def case_classifier_fitted_holdout(
    automl_cache: Callable[[str], AutoMLCache],
    dataset: str,
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a holdout fitted classifier"""
    resampling_strategy = "holdout-iterative-fit"

    cache = automl_cache(f"case_classifier_{resampling_strategy}_{dataset}")

    model = cache.model()
    if model is not None:
        return model

    X, y, Xt, yt = make_sklearn_dataset(name=dataset)

    model = make_automl_classifier(
        temporary_directory=cache.path("backend"),
        delete_tmp_folder_after_terminate=False,
        resampling_strategy=resampling_strategy,
    )
    model.fit(X, y, dataset_name=dataset)

    cache.save(model)
    return model


@case(tags=["classifier", "fitted", "cv"])
@parametrize("dataset", ["iris"])
def case_classifier_fitted_cv(
    automl_cache: Callable[[str], AutoMLCache],
    dataset: str,
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a fitted cv AutoMLClassifier"""
    resampling_strategy = "cv"
    cache = automl_cache(f"case_classifier_{resampling_strategy}_{dataset}")

    model = cache.model()
    if model is not None:
        return model

    X, y, Xt, yt = make_sklearn_dataset(name=dataset)
    model = make_automl_classifier(
        time_left_for_this_task=60,  # We include some extra time for cv
        per_run_time_limit=10,
        resampling_strategy=resampling_strategy,
        temporary_directory=cache.path("backend"),
        delete_tmp_folder_after_terminate=False,
    )
    model.fit(X, y, dataset_name=dataset)

    cache.save(model)
    return model


@case(tags=["regressor", "fitted", "holdout"])
@parametrize("dataset", ["boston"])
def case_regressor_fitted_holdout(
    automl_cache: Callable[[str], AutoMLCache],
    dataset: str,
    make_automl_regressor: Callable[..., AutoMLRegressor],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLRegressor:
    """Case of fitted regressor with cv resampling"""
    resampling_strategy = "holdout"
    cache = automl_cache(f"case_regressor_{resampling_strategy}_{dataset}")

    model = cache.model()
    if model is not None:
        return model

    X, y, Xt, yt = make_sklearn_dataset(name=dataset)
    model = make_automl_regressor(
        resampling_strategy=resampling_strategy,
        temporary_directory=cache.path("backend"),
        delete_tmp_folder_after_terminate=False,
    )
    model.fit(X, y, dataset_name=dataset)

    cache.save(model)
    return model


@case(tags=["regressor", "fitted", "cv"])
@parametrize("dataset", ["boston"])
def case_regressor_fitted_cv(
    automl_cache: Callable[[str], AutoMLCache],
    dataset: str,
    make_automl_regressor: Callable[..., AutoMLRegressor],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLRegressor:
    """Case of fitted regressor with cv resampling"""
    resampling_strategy = "cv"

    cache = automl_cache(f"case_regressor_{resampling_strategy}_{dataset}")
    model = cache.model()
    if model is not None:
        return model

    X, y, Xt, yt = make_sklearn_dataset(name=dataset)

    model = make_automl_regressor(
        time_left_for_this_task=60,
        per_run_time_limit=10,
        temporary_directory=cache.path("backend"),
        delete_tmp_folder_after_terminate=False,
        resampling_strategy=resampling_strategy,
    )
    model.fit(X, y, dataset_name=dataset)

    cache.save(model)
    return model


@case(tags=["classifier", "fitted", "no_ensemble"])
@parametrize("dataset", ["iris"])
def case_classifier_fitted_no_ensemble(
    automl_cache: Callable[[str], AutoMLCache],
    dataset: str,
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., Tuple[np.ndarray, ...]],
) -> AutoMLClassifier:
    """Case of a fitted classifier but enemble_size was set to 0"""
    cache = automl_cache(f"case_classifier_fitted_no_ensemble_{dataset}")

    model = cache.model()
    if model is not None:
        return model

    X, y, Xt, yt = make_sklearn_dataset(name=dataset)

    model = make_automl_classifier(
        temporary_directory=cache.path("backend"),
        delete_tmp_folder_after_terminate=False,
        ensemble_size=0,
    )
    model.fit(X, y, dataset_name=dataset)

    cache.save(model)
    return model
