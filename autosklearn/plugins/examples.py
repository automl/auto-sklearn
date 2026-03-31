"""
Example plugins demonstrating the plugin system.

This module contains example implementations of various plugin types.
"""

from typing import Any, Callable, Dict, List, Type

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from autosklearn.plugins import (
    ClassifierPlugin,
    MetricPlugin,
    OptimizerPlugin,
    PluginMetadata,
    PreprocessorPlugin,
    RegressorPlugin,
    register_plugin,
)


@register_plugin("classifier")
class CustomGradientBoostingClassifier(ClassifierPlugin):
    """Custom Gradient Boosting Classifier plugin."""

    metadata = PluginMetadata(
        name="custom_gradient_boosting",
        version="1.0.0",
        description="Custom Gradient Boosting Classifier with extended functionality",
        author="Auto-sklearn Team",
        tags=["tree", "ensemble", "boosting"],
    )

    @classmethod
    def get_name(cls) -> str:
        return "custom_gradient_boosting"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return cls.metadata

    @classmethod
    def get_estimator_class(cls) -> Type:
        return GradientBoostingClassifier

    @classmethod
    def get_properties(cls) -> Dict[str, Any]:
        return {
            "shortname": "CGB",
            "name": "Custom Gradient Boosting Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": ["dense"],
            "output": ["dense"],
        }


@register_plugin("regressor")
class CustomGradientBoostingRegressor(RegressorPlugin):
    """Custom Gradient Boosting Regressor plugin."""

    metadata = PluginMetadata(
        name="custom_gradient_boosting_regressor",
        version="1.0.0",
        description="Custom Gradient Boosting Regressor with extended functionality",
        author="Auto-sklearn Team",
        tags=["tree", "ensemble", "boosting", "regression"],
    )

    @classmethod
    def get_name(cls) -> str:
        return "custom_gradient_boosting_regressor"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return cls.metadata

    @classmethod
    def get_estimator_class(cls) -> Type:
        return GradientBoostingRegressor


@register_plugin("preprocessor")
class CustomStandardScaler(PreprocessorPlugin):
    """Custom Standard Scaler plugin."""

    metadata = PluginMetadata(
        name="custom_standard_scaler",
        version="1.0.0",
        description="Custom Standard Scaler with additional options",
        author="Auto-sklearn Team",
        tags=["scaling", "normalization"],
    )

    @classmethod
    def get_name(cls) -> str:
        return "custom_standard_scaler"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return cls.metadata

    @classmethod
    def get_transformer_class(cls) -> Type:
        return StandardScaler


@register_plugin("metric")
class F1BetaMetric(MetricPlugin):
    """F-beta score metric plugin."""

    metadata = PluginMetadata(
        name="f_beta",
        version="1.0.0",
        description="F-beta score metric with configurable beta",
        author="Auto-sklearn Team",
        tags=["metric", "classification", "f-score"],
    )

    _beta: float = 1.0

    @classmethod
    def get_name(cls) -> str:
        return "f_beta"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return cls.metadata

    @classmethod
    def get_metric_fn(cls) -> Callable:
        from sklearn.metrics import fbeta_score

        def f_beta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return fbeta_score(y_true, y_pred, beta=cls._beta, average="weighted")

        return f_beta

    @classmethod
    def get_sign(cls) -> int:
        return 1

    @classmethod
    def set_beta(cls, beta: float) -> None:
        """Set the beta parameter for F-beta score."""
        cls._beta = beta


@register_plugin("optimizer")
class RandomSearchOptimizer(OptimizerPlugin):
    """Random Search optimizer plugin."""

    metadata = PluginMetadata(
        name="random_search",
        version="1.0.0",
        description="Simple random search optimizer",
        author="Auto-sklearn Team",
        tags=["optimizer", "random", "baseline"],
    )

    @classmethod
    def get_name(cls) -> str:
        return "random_search"

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return cls.metadata

    @classmethod
    def get_optimizer_class(cls) -> Type:
        from sklearn.model_selection import RandomizedSearchCV

        return RandomizedSearchCV

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "n_iter": 10,
            "cv": 5,
            "random_state": 42,
        }


__all__ = [
    "CustomGradientBoostingClassifier",
    "CustomGradientBoostingRegressor",
    "CustomStandardScaler",
    "F1BetaMetric",
    "RandomSearchOptimizer",
]
