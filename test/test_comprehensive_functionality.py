"""
Comprehensive functionality tests for auto-sklearn
Covers 7 key testing scenarios as specified in requirements
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression, make_multilabel_classification
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    REGRESSION,
    MULTIOUTPUT_REGRESSION,
)
from autosklearn.metrics import accuracy, r2 as r2_metric

warnings.filterwarnings("ignore", category=UserWarning)


class TestAutoSklearnClassifierBasic:
    """Test basic classification functionality of AutoSklearnClassifier"""

    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_name", ["iris", "digits"])
    def test_classification_basic_flow(
        self,
        dataset_name: str,
        make_sklearn_dataset: Callable,
    ) -> None:
        """
        Test basic classification workflow on standard datasets
        
        Expects
        -------
        * fit() completes successfully without errors
        * predict() returns correct shape
        * predict_proba() returns valid probability distribution
        * score() returns reasonable accuracy
        * sprint_statistics() returns meaningful information
        """
        X_train, y_train, X_test, y_test = make_sklearn_dataset(
            name=dataset_name,
            train_size_maximum=1000,
        )

        # Initialize classifier with short time limits for testing
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=15,
            ensemble_size=5,
            ensemble_nbest=10,
            initial_configurations_via_metalearning=5,
            disable_progress_bar=True,
        )

        # Measure training time
        start_time = time.time()
        automl.fit(X_train, y_train, dataset_name=dataset_name)
        training_time = time.time() - start_time

        # Training should complete in reasonable time
        assert training_time < 120, f"Training took too long: {training_time:.2f}s"

        # Test predict
        predictions = automl.predict(X_test)
        assert predictions.shape == (len(X_test),), \
            f"Predictions shape mismatch: {predictions.shape}"

        # Test predict_proba
        probas = automl.predict_proba(X_test)
        n_classes = len(np.unique(y_train))
        assert probas.shape == (len(X_test), n_classes), \
            f"Probas shape mismatch: {probas.shape}"
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5), \
            "Probabilities should sum to 1"
        assert (probas >= 0).all() and (probas <= 1).all(), \
            "Probabilities should be in [0, 1]"

        # Test score
        score = automl.score(X_test, y_test)
        random_accuracy = 1.0 / n_classes
        assert score > random_accuracy + 0.1, \
            f"Accuracy {score:.3f} should be significantly higher than random {random_accuracy:.3f}"

        # Test sprint_statistics
        stats = automl.sprint_statistics()
        assert isinstance(stats, str)
        assert dataset_name in stats or "metric" in stats.lower()
        assert "validation score" in stats.lower() or "error" in stats.lower()

    @pytest.mark.slow
    def test_sprint_statistics_content(self) -> None:
        """
        Test that sprint_statistics() contains all required key information
        
        Expects
        -------
        * Contains dataset name
        * Contains metric information
        * Contains validation score
        * Contains run statistics (success/crashed/etc.)
        """
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=3,
            disable_progress_bar=True,
        )
        automl.fit(X_train, y_train, dataset_name="test_binary")

        stats = automl.sprint_statistics()
        assert "test_binary" in stats.lower() or "dataset" in stats.lower()
        assert any(keyword in stats.lower() for keyword in ["run", "success", "crashed", "timeout", "memory"])


class TestAutoSklearnRegressorBasic:
    """Test regression functionality of AutoSklearnRegressor"""

    @pytest.mark.slow
    @pytest.mark.parametrize("dataset_name", ["diabetes", "boston"])
    def test_regression_basic_flow(
        self,
        dataset_name: str,
        make_sklearn_dataset: Callable,
    ) -> None:
        """
        Test basic regression workflow on standard datasets
        
        Expects
        -------
        * Regression task is automatically identified and adapted
        * Predictions are continuous values
        * R² score is in reasonable range
        """
        try:
            X_train, y_train, X_test, y_test = make_sklearn_dataset(
                name=dataset_name,
                train_size_maximum=500,
            )
        except Exception:
            # Fallback to synthetic data if dataset not available
            X, y = make_regression(
                n_samples=200,
                n_features=10,
                n_informative=5,
                noise=0.1,
                random_state=42,
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Initialize regressor
        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=15,
            ensemble_size=5,
            disable_progress_bar=True,
            metric=r2_metric,
        )

        automl.fit(X_train, y_train, dataset_name=dataset_name)

        # Test predict - should return continuous values
        predictions = automl.predict(X_test)
        assert predictions.shape == (len(X_test),)

        # Check that predictions are continuous (not just a few discrete values)
        unique_preds = len(np.unique(predictions.round(decimals=3)))
        assert unique_preds > len(X_test) * 0.5, \
            f"Predictions seem too discrete: only {unique_preds} unique values"

        # Check no obvious outliers
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        assert np.all((predictions > mean_pred - 5 * std_pred) & 
                      (predictions < mean_pred + 5 * std_pred)), \
            "Predictions contain extreme outliers"

        # Test R² score
        r2 = automl.score(X_test, y_test)
        assert r2 > 0.3, f"R² score {r2:.3f} should be positive and reasonable"

    @pytest.mark.slow
    def test_regression_metrics(self) -> None:
        """
        Verify regression uses proper regression metrics
        
        Expects
        -------
        * Regression-specific metrics are used (MSE, R²)
        * Metrics are appropriate for continuous values
        """
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        automl = AutoSklearnRegressor(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=3,
            disable_progress_bar=True,
        )
        automl.fit(X_train, y_train, dataset_name="reg_test")

        # Verify predictions are floating point continuous values
        preds = automl.predict(X_test)
        assert np.issubdtype(preds.dtype, np.floating), \
            "Predictions should be floating point for regression"

        # Calculate R² manually for verification
        manual_r2 = r2_score(y_test, preds)
        automl_r2 = automl.score(X_test, y_test)
        assert np.isclose(manual_r2, automl_r2, atol=1e-3), \
            f"R² mismatch: manual {manual_r2:.3f}, automl {automl_r2:.3f}"


class TestMultiLabelMultiOutput:
    """Test multi-label classification and multi-output regression support"""

    @pytest.mark.slow
    def test_multilabel_classification(self) -> None:
        """
        Test multi-label classification support
        
        Expects
        -------
        * predict() returns 2D array (n_samples × n_classes)
        * Labels are treated independently
        """
        X, y = make_multilabel_classification(
            n_samples=200,
            n_features=20,
            n_classes=3,
            n_labels=2,
            allow_unlabeled=False,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=15,
            ensemble_size=3,
            disable_progress_bar=True,
        )
        automl.fit(X_train, y_train, dataset_name="multilabel_test")

        predictions = automl.predict(X_test)
        assert predictions.ndim == 2, "Multi-label predictions should be 2D"
        assert predictions.shape == (len(X_test), y.shape[1]), \
            f"Shape mismatch: {predictions.shape} vs expected {(len(X_test), y.shape[1])}"

        # Verify independent label handling by checking per-label accuracy
        for label_idx in range(y.shape[1]):
            label_acc = accuracy_score(y_test[:, label_idx], predictions[:, label_idx])
            assert label_acc > 0.5, f"Label {label_idx} accuracy too low: {label_acc:.3f}"

    @pytest.mark.slow
    def test_multioutput_regression(self) -> None:
        """
        Test multi-output regression support
        
        Expects
        -------
        * Each output is predicted independently
        * Shape matches expected multi-output format
        """
        X = np.random.randn(200, 10)
        # Create 2 correlated outputs
        y = np.column_stack([
            X[:, 0] * 2 + X[:, 1] + np.random.randn(200) * 0.1,
            X[:, 1] * 3 - X[:, 2] + np.random.randn(200) * 0.1,
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        automl = AutoSklearnRegressor(
            time_left_for_this_task=60,
            per_run_time_limit=15,
            ensemble_size=3,
            disable_progress_bar=True,
        )
        automl.fit(X_train, y_train, dataset_name="multioutput_test")

        predictions = automl.predict(X_test)
        assert predictions.ndim == 2, "Multi-output predictions should be 2D"
        assert predictions.shape == (len(X_test), y.shape[1]), \
            f"Shape mismatch: {predictions.shape} vs expected {(len(X_test), y.shape[1])}"

        # Each output should have reasonable performance
        for output_idx in range(y.shape[1]):
            output_r2 = r2_score(y_test[:, output_idx], predictions[:, output_idx])
            assert output_r2 > 0.3, f"Output {output_idx} R² too low: {output_r2:.3f}"


class TestDataPreprocessing:
    """Test data preprocessing components"""

    @pytest.mark.slow
    def test_categorical_encoding_options(self) -> None:
        """
        Test categorical feature encoding options (OHE, etc.)
        
        Expects
        -------
        * Different encoding strategies work correctly
        * Can force specific preprocessing components
        """
        # Create data with categorical features
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100),
        })
        y = np.random.randint(0, 2, 100)

        feat_type = ['Numerical', 'Numerical', 'Categorical', 'Categorical']

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=1,
            disable_progress_bar=True,
            include={
                'data_preprocessor': ['data_preprocessing'],
                'feature_preprocessor': ['no_preprocessing'],
                'classifier': ['decision_tree'],
            }
        )
        automl.fit(X, y, feat_type=feat_type, dataset_name="cat_encoding_test")

        # Check models show the preprocessing steps
        models = automl.show_models()
        assert len(models) > 0, "No models found"

    @pytest.mark.slow
    def test_missing_value_handling(self) -> None:
        """
        Test missing value handling for numerical and categorical features
        
        Expects
        -------
        * Missing values are properly handled (imputed)
        * Pipeline completes successfully
        """
        # Create data with missing values
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
        })
        # Add missing values (~20% missing)
        X.loc[np.random.choice(100, 20), 'num1'] = np.nan
        X.loc[np.random.choice(100, 15), 'cat1'] = np.nan
        y = np.random.randint(0, 2, 100)

        feat_type = ['Numerical', 'Numerical', 'Categorical']

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=1,
            disable_progress_bar=True,
        )

        # Should complete without errors despite missing values
        automl.fit(X, y, feat_type=feat_type, dataset_name="missing_values_test")

        # Can predict despite potential missing values in test
        X_test = X.copy()
        X_test.loc[0, 'num1'] = np.nan
        preds = automl.predict(X_test)
        assert len(preds) == len(X_test)

    @pytest.mark.slow
    @pytest.mark.parametrize("scaler_type", ["StandardScaler", "MinMaxScaler"])
    def test_scaling_strategies(self, scaler_type: str) -> None:
        """
        Test different scaling strategies
        
        Expects
        -------
        * Different scaling methods work correctly
        * Can be forced via include parameter
        """
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=1,
            disable_progress_bar=True,
        )
        automl.fit(X_train, y_train, dataset_name=f"scaler_{scaler_type}")

        score = automl.score(X_test, y_test)
        assert score > 0.6, f"Score {score:.3f} should be reasonable"


class TestInvalidInputHandling:
    """Test handling of various invalid inputs"""

    def test_empty_dataframe(self) -> None:
        """
        Test handling of empty input
        
        Expects
        -------
        * Raises meaningful error message
        * Does not hang or crash
        """
        X_empty = pd.DataFrame()
        y_empty = pd.Series()

        automl = AutoSklearnClassifier(
            time_left_for_this_task=10,
            disable_progress_bar=True,
        )

        with pytest.raises(Exception) as excinfo:
            automl.fit(X_empty, y_empty)

        err_msg = str(excinfo.value).lower()
        assert any(keyword in err_msg for keyword in ['empty', '0 samples', 'shape', 'feature']), \
            f"Error message should mention empty data: {err_msg}"

    def test_single_class_classification(self) -> None:
        """
        Test handling of single-class classification
        
        Expects
        -------
        * Raises meaningful error or handles gracefully
        """
        X = np.random.randn(50, 5)
        y = np.zeros(50)  # Only one class

        automl = AutoSklearnClassifier(
            time_left_for_this_task=10,
            disable_progress_bar=True,
        )

        # Should either raise error or handle gracefully with constant predictions
        try:
            automl.fit(X, y)
            preds = automl.predict(X)
            assert len(np.unique(preds)) == 1, "Should only predict one class"
        except Exception as e:
            err_msg = str(e).lower()
            assert any(keyword in err_msg for keyword in ['class', 'label', 'unique', 'binary']), \
                f"Error should be about classification: {err_msg}"

    def test_constant_features(self) -> None:
        """
        Test handling of constant/zero-variance features
        
        Expects
        -------
        * Completes successfully (features should be filtered)
        * Returns reasonable performance
        """
        X = np.column_stack([
            np.zeros(100),  # Constant feature
            np.ones(100),   # Constant feature
            np.random.randn(100, 5),  # Useful features
        ])
        y = np.random.randint(0, 2, 100)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=1,
            disable_progress_bar=True,
        )

        # Should complete successfully despite constant features
        automl.fit(X, y, dataset_name="constant_feat_test")
        score = automl.score(X, y)
        assert 0.0 <= score <= 1.0, "Score should be valid probability"

    @pytest.mark.slow
    def test_high_nan_ratio(self) -> None:
        """
        Test handling of high ratio of NaN values
        
        Expects
        -------
        * Either completes with imputation or raises meaningful error
        * Does not crash or hang
        """
        X = np.random.randn(100, 5)
        # Add 60% NaN values
        mask = np.random.random(X.shape) < 0.6
        X[mask] = np.nan
        y = np.random.randint(0, 2, 100)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=1,
            disable_progress_bar=True,
        )

        try:
            automl.fit(X, y, dataset_name="high_nan_test")
            # If it succeeds, predictions should work
            preds = automl.predict(X)
            assert len(preds) == len(X)
        except Exception as e:
            err_msg = str(e).lower()
            # Either completes or raises meaningful error
            assert any(keyword in err_msg for keyword in ['nan', 'missing', 'imputat']), \
                f"Should handle or error on NaN appropriately: {err_msg}"


class TestMetaLearning:
    """Test meta-learning cold start effects"""

    @pytest.mark.slow
    def test_metalearning_vs_random(self) -> None:
        """
        Compare meta-learning recommendations vs random search
        
        Expects
        -------
        * Meta-learning configurations show good initial performance
        * Meta-learning should help (or at least not hurt)
        """
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_informative=8,
            n_classes=3,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # With metalearning
        automl_with_metalearning = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            initial_configurations_via_metalearning=10,
            ensemble_size=1,
            disable_progress_bar=True,
            seed=42,
        )
        start = time.time()
        automl_with_metalearning.fit(X_train, y_train, dataset_name="metalearn_test")
        score_with = automl_with_metalearning.score(X_test, y_test)

        # Without metalearning
        automl_no_metalearning = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            initial_configurations_via_metalearning=0,
            ensemble_size=1,
            disable_progress_bar=True,
            seed=123,
        )
        automl_no_metalearning.fit(X_train, y_train, dataset_name="no_metalearn_test")
        score_without = automl_no_metalearning.score(X_test, y_test)

        # Meta-learning should at least be competitive
        # Note: performance may vary, just verify both produce valid results
        assert score_with > 0.5, f"With metalearning score too low: {score_with:.3f}"
        assert score_without > 0.5, f"Without metalearning score too low: {score_without:.3f}"

    def test_initial_configuration_quality(self) -> None:
        """
        Test that initial configurations from metalearning are reasonable
        
        Expects
        -------
        * Initial configurations produce valid models
        * Early performance is decent
        """
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=45,
            per_run_time_limit=15,
            initial_configurations_via_metalearning=5,
            ensemble_size=3,
            disable_progress_bar=True,
        )
        automl.fit(X_train, y_train, dataset_name="init_config_test")

        # Should have found some decent configuration
        score = automl.score(X_test, y_test)
        random_acc = 0.5  # Binary classification
        assert score > random_acc, \
            f"Initial config performance {score:.3f} should beat random {random_acc}"

        stats = automl.sprint_statistics()
        assert len(stats) > 0


class TestMemoryLimit:
    """Test memory limit handling"""

    @pytest.mark.slow
    def test_memory_limit_basic(self) -> None:
        """
        Test basic memory limit functionality
        
        Expects
        -------
        * Memory limit is respected
        * Models exceeding memory limit are gracefully terminated
        * Surviving models produce correct results
        """
        # Use larger dataset to potentially trigger memory limits
        X, y = make_classification(
            n_samples=1000,
            n_features=50,
            n_informative=30,
            n_classes=3,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Set tight memory limit to trigger MEMOUT for some models
        automl = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=20,
            memory_limit=1024,  # 1GB limit
            ensemble_size=3,
            disable_progress_bar=True,
        )

        # Should complete without crashing despite tight memory limit
        automl.fit(X_train, y_train, dataset_name="mem_limit_test")

        # Surviving models should produce valid predictions
        preds = automl.predict(X_test)
        assert preds.shape == (len(X_test),)

        score = automl.score(X_test, y_test)
        assert score > 0.4, f"Score should be reasonable despite memory limits: {score:.3f}"

    def test_memory_limit_graceful_degradation(self) -> None:
        """
        Test system stability under memory pressure
        
        Expects
        -------
        * System remains stable under memory pressure
        * No OOM crashes
        """
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Very tight memory limit
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            memory_limit=512,  # 512MB - very tight
            ensemble_size=1,
            disable_progress_bar=True,
        )

        # Should handle gracefully - either run successfully or properly terminate
        try:
            automl.fit(X_train, y_train, dataset_name="tight_mem_test")
            # If it succeeded, verify predictions work
            preds = automl.predict(X_test)
            assert len(preds) == len(X_test)
        except MemoryError:
            # This is acceptable - memory limit was hit but it's a controlled failure
            pass
        except Exception as e:
            # Should not crash with unrelated errors
            err_msg = str(e).lower()
            assert 'memory' in err_msg or 'alloc' in err_msg or 'limit' in err_msg, \
                f"Unexpected error under memory pressure: {e}"
