"""
Performance benchmark tests for Auto-sklearn.

These tests measure the performance of key components and track regressions.
Run with: pytest test/benchmark -v --benchmark-only
"""

import time
from typing import Any, Dict

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="module")
def classification_data() -> Dict[str, Any]:
    """Generate classification dataset for benchmarking."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    return {"X": X, "y": y}


@pytest.fixture(scope="module")
def regression_data() -> Dict[str, Any]:
    """Generate regression dataset for benchmarking."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42,
    )
    return {"X": X, "y": y}


class TestPipelinePerformance:
    """Benchmark tests for pipeline operations."""

    def test_pipeline_creation(self, benchmark):
        """Benchmark pipeline creation time."""
        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        def create_pipeline():
            return SimpleClassificationPipeline(
                config={"classifier:__choice__": "random_forest"},
                dataset_properties={"target_type": "classification"},
            )

        result = benchmark(create_pipeline)
        assert result is not None

    def test_pipeline_fit(self, benchmark, classification_data):
        """Benchmark pipeline fitting time."""
        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        pipeline = SimpleClassificationPipeline(
            config={"classifier:__choice__": "random_forest"},
            dataset_properties={"target_type": "classification"},
        )

        def fit_pipeline():
            pipeline.fit(classification_data["X"], classification_data["y"])

        benchmark(fit_pipeline)

    def test_pipeline_predict(self, benchmark, classification_data):
        """Benchmark pipeline prediction time."""
        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        pipeline = SimpleClassificationPipeline(
            config={"classifier:__choice__": "random_forest"},
            dataset_properties={"target_type": "classification"},
        )
        pipeline.fit(classification_data["X"], classification_data["y"])

        def predict():
            return pipeline.predict(classification_data["X"])

        result = benchmark(predict)
        assert len(result) == len(classification_data["y"])


class TestEnsemblePerformance:
    """Benchmark tests for ensemble operations."""

    def test_ensemble_building(self, benchmark, classification_data):
        """Benchmark ensemble building time."""
        from autosklearn.ensemble_building.builder import EnsembleBuilder

        def build_ensemble():
            builder = EnsembleBuilder(
                dataset_name="benchmark",
                task_type=1,
                metric="accuracy",
            )
            return builder

        result = benchmark(build_ensemble)
        assert result is not None


class TestDataValidationPerformance:
    """Benchmark tests for data validation."""

    def test_feature_validation(self, benchmark, classification_data):
        """Benchmark feature validation time."""
        from autosklearn.data.feature_validator import FeatureValidator

        validator = FeatureValidator()

        def validate():
            return validator.fit_transform(classification_data["X"])

        result = benchmark(validate)
        assert result is not None

    def test_target_validation(self, benchmark, classification_data):
        """Benchmark target validation time."""
        from autosklearn.data.target_validator import TargetValidator

        validator = TargetValidator(task_type="classification")

        def validate():
            return validator.fit_transform(classification_data["y"])

        result = benchmark(validate)
        assert result is not None


class TestMetalearningPerformance:
    """Benchmark tests for metalearning operations."""

    def test_metafeature_computation(self, benchmark, classification_data):
        """Benchmark metafeature computation time."""
        from autosklearn.metalearning.metafeatures.metafeatures import (
            calculate_all_metafeatures,
        )

        def compute_metafeatures():
            return calculate_all_metafeatures(
                classification_data["X"],
                classification_data["y"],
                task_type="classification",
            )

        result = benchmark(compute_metafeatures)
        assert result is not None


class TestMemoryUsage:
    """Memory usage benchmarks."""

    @pytest.mark.skip(reason="Requires memory_profiler")
    def test_memory_pipeline_fit(self, classification_data):
        """Test memory usage during pipeline fitting."""
        from memory_profiler import memory_usage

        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        def fit_pipeline():
            pipeline = SimpleClassificationPipeline(
                config={"classifier:__choice__": "random_forest"},
                dataset_properties={"target_type": "classification"},
            )
            pipeline.fit(classification_data["X"], classification_data["y"])

        mem_usage = memory_usage(fit_pipeline, max_usage=True)
        assert mem_usage < 500  # MB


class TestScalability:
    """Scalability benchmarks for different data sizes."""

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 5000])
    def test_scaling_fit_time(self, benchmark, n_samples):
        """Test how fit time scales with dataset size."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            random_state=42,
        )

        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        pipeline = SimpleClassificationPipeline(
            config={"classifier:__choice__": "random_forest"},
            dataset_properties={"target_type": "classification"},
        )

        def fit():
            pipeline.fit(X, y)

        benchmark(fit)

    @pytest.mark.parametrize("n_features", [10, 50, 100, 200])
    def test_scaling_features(self, benchmark, n_features):
        """Test how fit time scales with number of features."""
        X, y = make_classification(
            n_samples=1000,
            n_features=n_features,
            n_informative=n_features // 2,
            random_state=42,
        )

        from autosklearn.pipeline.classification import SimpleClassificationPipeline

        pipeline = SimpleClassificationPipeline(
            config={"classifier:__choice__": "random_forest"},
            dataset_properties={"target_type": "classification"},
        )

        def fit():
            pipeline.fit(X, y)

        benchmark(fit)
