"""
Plugin system for Auto-sklearn.

This module provides a plugin architecture that allows:
- Dynamic loading of new machine learning algorithms
- Custom hyperparameter optimization strategies
- Extensible evaluation metrics
- Third-party integrations

Usage:
    from autosklearn.plugins import PluginRegistry, register_plugin

    @register_plugin('classifier')
    class MyCustomClassifier:
        ...
"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class PluginType(Enum):
    """Types of plugins supported by Auto-sklearn."""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    PREPROCESSOR = "preprocessor"
    DATA_PREPROCESSOR = "data_preprocessor"
    FEATURE_PREPROCESSOR = "feature_preprocessor"
    METRIC = "metric"
    OPTIMIZER = "optimizer"
    ENSEMBLE = "ensemble"
    VALIDATOR = "validator"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    email: str = ""
    url: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)


class PluginBase(ABC):
    """Base class for all plugins."""

    metadata: PluginMetadata

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the plugin name."""
        pass

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @classmethod
    def get_config_space(cls) -> Any:
        """Return the configuration space for this plugin.

        Returns:
            ConfigurationSpace: The hyperparameter configuration space
        """
        return None

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate a configuration for this plugin.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        return True


class ClassifierPlugin(PluginBase):
    """Base class for classifier plugins."""

    @classmethod
    @abstractmethod
    def get_estimator_class(cls) -> Type:
        """Return the sklearn-compatible estimator class."""
        pass

    @classmethod
    def get_properties(cls) -> Dict[str, Any]:
        """Return properties of the classifier.

        Returns:
            Dictionary with properties like:
            - shortname: Short name for the classifier
            - name: Full name
            - handles_regression: bool
            - handles_classification: bool
            - handles_multiclass: bool
            - handles_multilabel: bool
            - handles_multioutput: bool
            - is_deterministic: bool
            - input: List of accepted data types
            - output: List of output types
        """
        return {
            "shortname": cls.get_name(),
            "name": cls.get_name(),
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": ["dense", "sparse"],
            "output": ["dense"],
        }


class RegressorPlugin(PluginBase):
    """Base class for regressor plugins."""

    @classmethod
    @abstractmethod
    def get_estimator_class(cls) -> Type:
        """Return the sklearn-compatible estimator class."""
        pass

    @classmethod
    def get_properties(cls) -> Dict[str, Any]:
        """Return properties of the regressor."""
        return {
            "shortname": cls.get_name(),
            "name": cls.get_name(),
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": ["dense", "sparse"],
            "output": ["dense"],
        }


class PreprocessorPlugin(PluginBase):
    """Base class for preprocessor plugins."""

    @classmethod
    @abstractmethod
    def get_transformer_class(cls) -> Type:
        """Return the sklearn-compatible transformer class."""
        pass

    @classmethod
    def get_properties(cls) -> Dict[str, Any]:
        """Return properties of the preprocessor."""
        return {
            "shortname": cls.get_name(),
            "name": cls.get_name(),
            "input": ["dense", "sparse"],
            "output": ["dense", "sparse"],
        }


class MetricPlugin(PluginBase):
    """Base class for custom metric plugins."""

    @classmethod
    @abstractmethod
    def get_metric_fn(cls) -> Callable:
        """Return the metric function."""
        pass

    @classmethod
    @abstractmethod
    def get_sign(cls) -> int:
        """Return the sign of the metric.

        Returns:
            1 for metrics to maximize, -1 for metrics to minimize
        """
        pass

    @classmethod
    def get_optimum(cls) -> float:
        """Return the optimum value for this metric."""
        if cls.get_sign() == 1:
            return 1.0
        else:
            return 0.0

    @classmethod
    def get_range(cls) -> tuple:
        """Return the range of possible values."""
        return (0.0, 1.0)


class OptimizerPlugin(PluginBase):
    """Base class for hyperparameter optimizer plugins."""

    @classmethod
    @abstractmethod
    def get_optimizer_class(cls) -> Type:
        """Return the optimizer class."""
        pass

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration for the optimizer."""
        return {}


class PluginRegistry:
    """Registry for managing plugins.

    This class provides a centralized registry for all plugins in Auto-sklearn.
    It supports dynamic loading, registration, and retrieval of plugins.

    Example:
        registry = PluginRegistry()

        @registry.register('classifier')
        class MyClassifier(ClassifierPlugin):
            ...

        classifier = registry.get('classifier', 'my_classifier')
    """

    _instance: Optional["PluginRegistry"] = None

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins: Dict[PluginType, Dict[str, Type[PluginBase]]] = {
                pt: {} for pt in PluginType
            }
            cls._instance._hooks: Dict[str, List[Callable]] = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PluginRegistry":
        """Get the singleton instance of the registry."""
        return cls()

    def register(
        self, plugin_type: Union[PluginType, str]
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a plugin.

        Args:
            plugin_type: Type of plugin to register

        Returns:
            Decorator function

        Example:
            @registry.register('classifier')
            class MyClassifier(ClassifierPlugin):
                ...
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type)

        def decorator(cls: Type[T]) -> Type[T]:
            name = cls.get_name()
            self._plugins[plugin_type][name] = cls
            return cls

        return decorator

    def register_plugin(
        self, plugin_type: Union[PluginType, str], plugin: Type[PluginBase]
    ) -> None:
        """Register a plugin directly.

        Args:
            plugin_type: Type of plugin
            plugin: Plugin class to register
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type)
        name = plugin.get_name()
        self._plugins[plugin_type][name] = plugin

    def get(
        self, plugin_type: Union[PluginType, str], name: str
    ) -> Optional[Type[PluginBase]]:
        """Get a registered plugin.

        Args:
            plugin_type: Type of plugin
            name: Name of the plugin

        Returns:
            Plugin class or None if not found
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type)
        return self._plugins[plugin_type].get(name)

    def get_all(self, plugin_type: Union[PluginType, str]) -> Dict[str, Type[PluginBase]]:
        """Get all registered plugins of a type.

        Args:
            plugin_type: Type of plugin

        Returns:
            Dictionary of plugin name to plugin class
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type)
        return self._plugins[plugin_type].copy()

    def list_plugins(self, plugin_type: Union[PluginType, str]) -> List[str]:
        """List all registered plugin names of a type.

        Args:
            plugin_type: Type of plugin

        Returns:
            List of plugin names
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type)
        return list(self._plugins[plugin_type].keys())

    def unregister(self, plugin_type: Union[PluginType, str], name: str) -> bool:
        """Unregister a plugin.

        Args:
            plugin_type: Type of plugin
            name: Name of the plugin

        Returns:
            True if plugin was removed, False if not found
        """
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type)
        if name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]
            return True
        return False

    def load_from_entry_points(self) -> None:
        """Load plugins from entry points.

        This method loads plugins that have been registered via the
        poetry.plugins or setuptools entry_points mechanism.
        """
        try:
            import importlib.metadata as importlib_metadata
        except ImportError:
            import importlib_metadata

        for plugin_type in PluginType:
            entry_point_group = f"autosklearn.{plugin_type.value}s"
            try:
                entry_points = importlib_metadata.entry_points(group=entry_point_group)
                for ep in entry_points:
                    try:
                        plugin_class = ep.load()
                        self.register_plugin(plugin_type, plugin_class)
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"Failed to load plugin {ep.name}: {e}",
                            RuntimeWarning,
                        )
            except Exception:
                pass

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a callback hook for an event.

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def trigger_hook(self, event: str, *args, **kwargs) -> None:
        """Trigger all callbacks for an event.

        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        for callback in self._hooks.get(event, []):
            callback(*args, **kwargs)


def register_plugin(plugin_type: Union[PluginType, str]) -> Callable[[Type[T]], Type[T]]:
    """Convenience function to register a plugin.

    Args:
        plugin_type: Type of plugin to register

    Returns:
        Decorator function

    Example:
        @register_plugin('classifier')
        class MyClassifier(ClassifierPlugin):
            ...
    """
    registry = PluginRegistry.get_instance()
    return registry.register(plugin_type)


def get_plugin(plugin_type: Union[PluginType, str], name: str) -> Optional[Type[PluginBase]]:
    """Convenience function to get a registered plugin.

    Args:
        plugin_type: Type of plugin
        name: Name of the plugin

    Returns:
        Plugin class or None if not found
    """
    registry = PluginRegistry.get_instance()
    return registry.get(plugin_type, name)


__all__ = [
    "PluginType",
    "PluginMetadata",
    "PluginBase",
    "ClassifierPlugin",
    "RegressorPlugin",
    "PreprocessorPlugin",
    "MetricPlugin",
    "OptimizerPlugin",
    "PluginRegistry",
    "register_plugin",
    "get_plugin",
]
