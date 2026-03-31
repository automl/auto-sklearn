"""Benchmark tests for classification tasks."""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import autosklearn.classification


class TestClassificationBenchmarks:
    """Benchmark tests for classification tasks."""

    @pytest.fixture(scope="class")
    def small_classification_data(self):
        """Generate a small classification dataset for quick benchmarks."""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    @pytest.fixture(scope="class")
    def medium_classification_data(self):
        """Generate a medium classification dataset for more realistic benchmarks."""
        X, y = make_classification(
            n_samples=10000,
            n_features=50,
            n_informative=25,
            n_redundant=10,
            n_classes=5,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_classifier_fit_small(self, benchmark, small_classification_data):
        """Benchmark fitting a classifier on a small dataset."""
        X_train, X_test, y_train, y_test = small_classification_data
        
        def fit_classifier():
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=30,
                per_run_time_limit=10,
                ensemble_size=5,
                initial_configurations_via_metalearning=0,
                random_state=42,
            )
            automl.fit(X_train, y_train, dataset_name="small_classification")
            return automl
        
        result = benchmark(fit_classifier)
        assert result is not None
        assert hasattr(result, "predict")

    def test_classifier_predict_small(self, benchmark, small_classification_data):
        """Benchmark predicting with a fitted classifier on a small dataset."""
        X_train, X_test, y_train, y_test = small_classification_data
        
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            ensemble_size=5,
            initial_configurations_via_metalearning=0,
            random_state=42,
        )
        automl.fit(X_train, y_train, dataset_name="small_classification")
        
        def predict():
            return automl.predict(X_test)
        
        predictions = benchmark(predict)
        assert len(predictions) == len(y_test)
        assert np.all(np.isfinite(predictions))

    def test_classifier_fit_medium(self, benchmark, medium_classification_data):
        """Benchmark fitting a classifier on a medium dataset."""
        X_train, X_test, y_train, y_test = medium_classification_data
        
        def fit_classifier():
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=60,
                per_run_time_limit=20,
                ensemble_size=5,
                initial_configurations_via_metalearning=0,
                random_state=42,
            )
            automl.fit(X_train, y_train, dataset_name="medium_classification")
            return automl
        
        result = benchmark(fit_classifier)
        assert result is not None
        assert hasattr(result, "predict")

    def test_classifier_predict_medium(self, benchmark, medium_classification_data):
        """Benchmark predicting with a fitted classifier on a medium dataset."""
        X_train, X_test, y_train, y_test = medium_classification_data
        
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=20,
            ensemble_size=5,
            initial_configurations_via_metalearning=0,
            random_state=42,
        )
        automl.fit(X_train, y_train, dataset_name="medium_classification")
        
        def predict():
            return automl.predict(X_test)
        
        predictions = benchmark(predict)
        assert len(predictions) == len(y_test)
        assert np.all(np.isfinite(predictions))
