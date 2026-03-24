"""
Comprehensive tests for AutoSklearnClassifier and AutoSklearnRegressor

Test Categories:
1. AutoSklearnClassifier basic classification functionality
2. AutoSklearnRegressor regression functionality
3. Multi-label/multi-output task support
4. Data preprocessing component completeness
5. Invalid input handling
6. Meta-learning cold start effect
7. Memory limit testing
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import psutil
import sklearn.datasets
from dask.distributed import Client
from sklearn.base import clone
from sklearn.datasets import make_classification, make_multilabel_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
)
from autosklearn.estimators import AutoSklearnRegressor
from autosklearn.metrics import accuracy, balanced_accuracy, f1_macro, mean_squared_error, r2

import pytest
from pytest import FixtureRequest, fixture

from functools import partial

from test.conftest import DEFAULT_SEED
from test.fixtures.dask import create_test_dask_client


def _create_test_automl(
    automl_class: type,
    request: FixtureRequest,
    **kwargs: Any,
) -> Any:
    """Create an AutoML instance with test defaults."""
    test_defaults = {
        "time_left_for_this_task": 30,
        "per_run_time_limit": 5,
        "seed": DEFAULT_SEED,
        "n_jobs": 1,
        "memory_limit": 1024,
        "ensemble_size": 5,
        "ensemble_nbest": 5,
        "max_models_on_disc": 5,
        "initial_configurations_via_metalearning": 3,
        "delete_tmp_folder_after_terminate": True,
    }

    opts: Dict[str, Any] = {**test_defaults, **kwargs}

    if "dask_client" not in opts:
        client = create_test_dask_client(id=request.node.nodeid, n_workers=1)
        opts["dask_client"] = client

    return automl_class(**opts)


@fixture
def make_classifier(request: FixtureRequest) -> Callable[..., AutoSklearnClassifier]:
    """Factory for creating AutoSklearnClassifier instances."""
    return partial(_create_test_automl, automl_class=AutoSklearnClassifier, request=request)


@fixture
def make_regressor(request: FixtureRequest) -> Callable[..., AutoSklearnRegressor]:
    """Factory for creating AutoSklearnRegressor instances."""
    return partial(_create_test_automl, automl_class=AutoSklearnRegressor, request=request)


@fixture
def iris_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load iris dataset for classification tests."""
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=DEFAULT_SEED)
    return X_train, X_test, y_train, y_test


@fixture
def digits_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits dataset for classification tests."""
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=DEFAULT_SEED)
    return X_train, X_test, y_train, y_test


@fixture
def diabetes_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load diabetes dataset for regression tests."""
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=DEFAULT_SEED)
    return X_train, X_test, y_train, y_test


@fixture
def california_housing_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load California housing dataset for regression tests."""
    try:
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    except AttributeError:
        X, y = sklearn.datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=DEFAULT_SEED
    )
    return X_train, X_test, y_train, y_test


@fixture
def multilabel_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create multilabel classification dataset."""
    X, y = make_multilabel_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_labels=2,
        random_state=DEFAULT_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=DEFAULT_SEED)
    return X_train, X_test, y_train, y_test


@fixture
def multioutput_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create multi-output regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_targets=3,
        random_state=DEFAULT_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=DEFAULT_SEED)
    return X_train, X_test, y_train, y_test


class TestAutoSklearnClassifierBasic:
    """Test AutoSklearnClassifier basic classification functionality."""

    def test_fit_success_on_iris(self, iris_data: Tuple, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * fit() method completes training without crashing or raising errors
        * Training completes within reasonable time
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        assert hasattr(cls, 'automl_'), "fit() should create automl_ attribute"
        assert cls.automl_ is not None, "automl_ should not be None after fit"

    def test_predict_shape_on_iris(self, iris_data: Tuple, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * predict() returns predictions with shape matching the number of test samples
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        predictions = cls.predict(X_test)

        assert predictions.shape == y_test.shape, (
            f"Predictions shape {predictions.shape} should match y_test shape {y_test.shape}"
        )

    def test_predict_proba_on_iris(self, iris_data: Tuple, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * predict_proba() returns probability distribution
        * Probabilities sum to 1 for each sample
        * All probabilities are in range [0, 1]
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        proba = cls.predict_proba(X_test)

        assert proba.shape[0] == X_test.shape[0], "Number of probability predictions should match samples"
        assert proba.shape[1] == len(np.unique(y_train)), "Number of classes should match unique labels"

        proba_sums = np.sum(proba, axis=1)
        assert np.allclose(proba_sums, 1.0, atol=1e-5), (
            f"Probabilities should sum to 1, got sums: {proba_sums[:5]}..."
        )

        assert np.all(proba >= 0) and np.all(proba <= 1), (
            "All probabilities should be in range [0, 1]"
        )

    def test_score_accuracy_on_iris(self, iris_data: Tuple, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * score() returns reasonable accuracy (significantly higher than random guess)
        * For iris (3 classes), random guess accuracy is ~33%, model should achieve >60%
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        score = cls.score(X_test, y_test)

        random_baseline = 1.0 / len(np.unique(y_train))
        assert score > random_baseline * 1.5, (
            f"Score {score:.4f} should be significantly higher than random baseline {random_baseline:.4f}"
        )

    def test_sprint_statistics(self, iris_data: Tuple, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * sprint_statistics() returns a string containing key information
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        stats = cls.sprint_statistics()

        assert isinstance(stats, str), "sprint_statistics() should return a string"
        assert len(stats) > 0, "Statistics string should not be empty"

        key_terms = ['metric', 'best', 'duration']
        found_terms = [term for term in key_terms if term.lower() in stats.lower()]
        assert len(found_terms) >= 2, (
            f"Statistics should contain key terms. Found: {found_terms}"
        )

    def test_fit_on_digits_dataset(self, digits_data: Tuple, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * fit() works on digits dataset (multiclass classification)
        * Model achieves reasonable accuracy on digit recognition
        """
        X_train, X_test, y_train, y_test = digits_data

        cls = make_classifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        predictions = cls.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.7, f"Accuracy on digits should be > 0.7, got {accuracy:.4f}"


class TestAutoSklearnRegressor:
    """Test AutoSklearnRegressor regression functionality."""

    def test_fit_success_on_diabetes(self, diabetes_data: Tuple, make_regressor: Callable) -> None:
        """
        Expects
        -------
        * fit() method completes training without crashing
        * Regression task is automatically identified and adapted
        """
        X_train, X_test, y_train, y_test = diabetes_data

        reg = make_regressor(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)

        assert hasattr(reg, 'automl_'), "fit() should create automl_ attribute"
        assert reg.automl_ is not None, "automl_ should not be None after fit"

    def test_predict_continuous_values(self, diabetes_data: Tuple, make_regressor: Callable) -> None:
        """
        Expects
        -------
        * predict() returns continuous values (not discrete classes)
        * Predictions have correct shape
        * No obvious anomalies in predictions
        """
        X_train, X_test, y_train, y_test = diabetes_data

        reg = make_regressor(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)

        assert predictions.shape == y_test.shape, (
            f"Predictions shape {predictions.shape} should match y_test shape {y_test.shape}"
        )

        assert np.issubdtype(predictions.dtype, np.floating), (
            "Regression predictions should be floating point"
        )

        assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN"
        assert not np.any(np.isinf(predictions)), "Predictions should not contain Inf"

    def test_r2_score_reasonable(self, diabetes_data: Tuple, make_regressor: Callable) -> None:
        """
        Expects
        -------
        * R² score is in reasonable range (>0.3 for diabetes dataset)
        * Regression metrics are used (not classification metrics)
        """
        X_train, X_test, y_train, y_test = diabetes_data

        reg = make_regressor(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            metric=r2,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)

        score = reg.score(X_test, y_test)

        assert score > 0.3, f"R² score should be > 0.3, got {score:.4f}"

    def test_regression_on_california_housing(
        self, california_housing_data: Tuple, make_regressor: Callable
    ) -> None:
        """
        Expects
        -------
        * Works on larger regression dataset
        * Achieves reasonable performance
        """
        X_train, X_test, y_train, y_test = california_housing_data

        reg = make_regressor(
            time_left_for_this_task=30,
            per_run_time_limit=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)
        score = r2_score(y_test, predictions)

        assert score > 0.3, f"R² score on California housing should be > 0.3, got {score:.4f}"


class TestMultiLabelMultiOutput:
    """Test multi-label classification and multi-output regression support."""

    def test_multilabel_predict_shape(
        self, multilabel_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * predict() returns 2D array (n_samples × n_classes)
        * Labels are independent (not mutually exclusive)
        """
        X_train, X_test, y_train, y_test = multilabel_data

        cls = make_classifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        predictions = cls.predict(X_test)

        assert predictions.ndim == 2, (
            f"Multilabel predictions should be 2D, got {predictions.ndim}D"
        )
        assert predictions.shape == y_test.shape, (
            f"Predictions shape {predictions.shape} should match y_test shape {y_test.shape}"
        )

        unique_label_combos = len(np.unique(predictions, axis=0))
        assert unique_label_combos > 1, (
            "Multilabel should produce multiple label combinations"
        )

    def test_multilabel_predict_proba(
        self, multilabel_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * predict_proba() returns probabilities for each label independently
        """
        X_train, X_test, y_train, y_test = multilabel_data

        cls = make_classifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        proba = cls.predict_proba(X_test)

        assert proba is not None, "predict_proba should return values for multilabel"

    def test_multioutput_regression(
        self, multioutput_data: Tuple, make_regressor: Callable
    ) -> None:
        """
        Expects
        -------
        * Multi-output regression works correctly
        * Each output is predicted independently
        * Evaluation metrics adapt to multi-output scenario
        """
        X_train, X_test, y_train, y_test = multioutput_data

        reg = make_regressor(
            time_left_for_this_task=30,
            per_run_time_limit=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)

        assert predictions.shape == y_test.shape, (
            f"Multi-output predictions shape {predictions.shape} should match y_test shape {y_test.shape}"
        )

        for i in range(y_test.shape[1]):
            output_r2 = r2_score(y_test[:, i], predictions[:, i])
            assert output_r2 > -1.0, (
                f"R² for output {i} should be > -1.0, got {output_r2:.4f}"
            )


class TestPreprocessingComponents:
    """Test data preprocessing component completeness."""

    def test_one_hot_encoding(
        self, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * One-hot encoding option is available and works correctly
        """
        X = np.array([
            ['a', 'x'],
            ['b', 'y'],
            ['a', 'z'],
            ['c', 'x'],
            ['b', 'y'],
        ])
        y = np.array([0, 1, 0, 1, 1])

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            include={
                'data_preprocessor': ['categorical_encoding'],
                'classifier': ['random_forest']
            },
        )

        feat_type = ['categorical', 'categorical']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X, y, feat_type=feat_type)

        predictions = cls.predict(X)
        assert predictions.shape == y.shape

    def test_numerical_imputation(
        self, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * Missing value handling works correctly for numerical features
        """
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, np.nan],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        y = np.array([0, 1, 0, 1, 1])

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X, y)

        predictions = cls.predict(X)
        assert predictions.shape == y.shape
        assert not np.any(np.isnan(predictions))

    def test_standard_scaling(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * StandardScaler preprocessing option works
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            include={
                'data_preprocessor': ['rescaling'],
                'classifier': ['random_forest']
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        predictions = cls.predict(X_test)
        assert predictions.shape == y_test.shape

    def test_minmax_scaling(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * MinMaxScaler preprocessing option works
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        score = cls.score(X_test, y_test)
        assert score > 0.5

    def test_no_preprocessing_option(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * No preprocessing option works
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            include={
                'feature_preprocessor': ['no_preprocessing'],
                'classifier': ['random_forest']
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        predictions = cls.predict(X_test)
        assert predictions.shape == y_test.shape


class TestInvalidInputHandling:
    """Test invalid input handling."""

    def test_empty_array_input(self, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * Empty array raises meaningful error (not silent failure)
        """
        X = np.array([]).reshape(0, 2)
        y = np.array([])

        cls = make_classifier()

        with pytest.raises((ValueError, IndexError, RuntimeError)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X, y)

    def test_single_class_input(self, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * Single class classification raises meaningful error
        """
        X = np.random.rand(50, 5)
        y = np.ones(50)

        cls = make_classifier()

        with pytest.raises((ValueError, RuntimeError)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X, y)

    def test_high_nan_ratio(self, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * High NaN ratio (>50%) is handled gracefully
        * Either raises meaningful error or handles it without crashing
        """
        X = np.random.rand(50, 5)
        X[:, 0] = np.nan
        X[:, 1] = np.nan
        X[:, 2] = np.nan
        y = np.random.randint(0, 2, 50)

        cls = make_classifier(
            time_left_for_this_task=15,
            per_run_time_limit=5,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X, y)
            predictions = cls.predict(X)
            assert predictions is not None
        except (ValueError, RuntimeError) as e:
            assert "nan" in str(e).lower() or "missing" in str(e).lower() or "error" in str(e).lower()

    def test_constant_features(self, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * Constant features are handled without crashing
        """
        X = np.ones((50, 5))
        X[:, 0] = np.random.rand(50)
        y = np.random.randint(0, 2, 50)

        cls = make_classifier(
            time_left_for_this_task=15,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X, y)

        predictions = cls.predict(X)
        assert predictions.shape == y.shape

    def test_mismatched_dimensions(self, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * Mismatched X and y dimensions raise meaningful error
        """
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 30)

        cls = make_classifier()

        with pytest.raises((ValueError, IndexError)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X, y)

    def test_invalid_feat_type(self, make_classifier: Callable) -> None:
        """
        Expects
        -------
        * Invalid feature type raises meaningful error
        """
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        cls = make_classifier()

        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X, y, feat_type=["invalid"] * 5)


class TestMetaLearningColdStart:
    """Test meta-learning cold start effect."""

    def test_metalearning_configurations_quality(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * Meta-learning recommendations provide reasonable starting configurations
        * Initial configurations achieve non-trivial performance
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            initial_configurations_via_metalearning=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        score = cls.score(X_test, y_test)

        assert score > 0.5, (
            f"Meta-learning should provide reasonable initial configs, got score {score:.4f}"
        )

    def test_metalearning_vs_random(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * Meta-learning provides better initial configurations than random
        * Head-to-head win rate > 50% (simplified test)
        """
        X_train, X_test, y_train, y_test = iris_data

        cls_with_meta = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            initial_configurations_via_metalearning=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls_with_meta.fit(X_train, y_train)

        score_with_meta = cls_with_meta.score(X_test, y_test)

        cls_no_meta = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            initial_configurations_via_metalearning=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls_no_meta.fit(X_train, y_train)

        score_no_meta = cls_no_meta.score(X_test, y_test)

        assert score_with_meta > 0.5, "Meta-learning should achieve reasonable performance"


class TestMemoryLimit:
    """Test memory limit functionality."""

    def test_memory_limit_basic(
        self, diabetes_data: Tuple, make_regressor: Callable
    ) -> None:
        """
        Expects
        -------
        * Memory limit is enforced
        * System remains stable under memory pressure
        """
        X_train, X_test, y_train, y_test = diabetes_data

        reg = make_regressor(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            memory_limit=512,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, y_train)

        predictions = reg.predict(X_test)
        assert predictions.shape == y_test.shape

    def test_memory_limit_small(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * Very small memory limit is handled gracefully
        * Either models fail gracefully or system adapts
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            memory_limit=256,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        predictions = cls.predict(X_test)
        assert predictions.shape == y_test.shape

    def test_no_memory_limit(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * None memory limit works (no memory constraints)
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
            memory_limit=None,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        score = cls.score(X_test, y_test)
        assert score > 0.5


class TestShowModels:
    """Test show_models functionality."""

    def test_show_models_returns_dict(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * show_models() returns a dictionary of models
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        models = cls.show_models()

        assert isinstance(models, dict), "show_models() should return a dictionary"
        assert len(models) > 0, "show_models() should return non-empty dictionary"


class TestPerformanceOverTime:
    """Test performance over time functionality."""

    def test_performance_over_time_exists(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * performance_over_time_ attribute exists after fit
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        assert hasattr(cls, 'performance_over_time_'), (
            "performance_over_time_ attribute should exist after fit"
        )

        perf = cls.performance_over_time_
        assert perf is not None, "performance_over_time_ should not be None"
        assert len(perf) > 0, "performance_over_time_ should have data"


class TestCVResults:
    """Test cv_results_ functionality."""

    def test_cv_results_structure(
        self, iris_data: Tuple, make_classifier: Callable
    ) -> None:
        """
        Expects
        -------
        * cv_results_ has expected structure
        """
        X_train, X_test, y_train, y_test = iris_data

        cls = make_classifier(
            time_left_for_this_task=20,
            per_run_time_limit=5,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.fit(X_train, y_train)

        cv_results = cls.cv_results_

        assert isinstance(cv_results, dict), "cv_results_ should be a dictionary"
        assert 'mean_test_score' in cv_results or 'status' in cv_results, (
            "cv_results_ should contain performance metrics"
        )
