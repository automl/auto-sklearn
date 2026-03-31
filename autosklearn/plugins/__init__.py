"""Plugin system for auto-sklearn.

This module provides a plugin architecture that allows extending auto-sklearn with:
- New machine learning algorithms
- Custom hyperparameter optimization strategies
- Extensible evaluation metrics
- Third-party integrations

Examples:
    >>> from autosklearn.plugins import PluginManager
    >>> manager = PluginManager()
    >>> manager.load_plugins()
    >>> available_algorithms = manager.get_plugins('algorithm')
"""

import importlib
import importlib.util
import inspect
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

try:
    from ConfigSpace.configuration_space import ConfigurationSpace
except ImportError:
    ConfigurationSpace = None


class PluginType(Enum):
    """Types of plugins that can be loaded."""
    ALGORITHM = auto()
    PREPROCESSING = auto()
    HPO_STRATEGY = auto()
    METRIC = auto()
    ENSEMBLE_METHOD = auto()
    CALLBACK = auto()
    DATA_LOADER = auto()


class Plugin(ABC):
    """Base class for all plugins."""
    
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the unique name of the plugin."""
        pass
    
    @classmethod
    @abstractmethod
    def get_type(cls) -> PluginType:
        """Return the type of the plugin."""
        pass
    
    @classmethod
    def get_description(cls) -> str:
        """Return a description of the plugin."""
        return cls.__doc__ or ""
    
    @classmethod
    def get_version(cls) -> str:
        """Return the version of the plugin."""
        return "1.0.0"
    
    @classmethod
    def get_configuration_space(cls) -> Optional[ConfigurationSpace]:
        """Return the configuration space for hyperparameter optimization."""
        return None


class AlgorithmPlugin(Plugin):
    """Base class for algorithm plugins."""
    
    @classmethod
    def get_type(cls) -> PluginType:
        return PluginType.ALGORITHM
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Fit the algorithm to the data."""
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """Make predictions with the fitted algorithm."""
        pass


class PreprocessingPlugin(Plugin):
    """Base class for preprocessing plugins."""
    
    @classmethod
    def get_type(cls) -> PluginType:
        return PluginType.PREPROCESSING
    
    @abstractmethod
    def fit_transform(self, X, y=None, **kwargs):
        """Fit the preprocessor to the data and transform it."""
        pass
    
    @abstractmethod
    def transform(self, X, **kwargs):
        """Transform the data using the fitted preprocessor."""
        pass


class MetricPlugin(Plugin):
    """Base class for metric plugins."""
    
    @classmethod
    def get_type(cls) -> PluginType:
        return PluginType.METRIC
    
    @abstractmethod
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        """Compute the metric value."""
        pass
    
    @classmethod
    def is_better(cls, value1: float, value2: float) -> bool:
        """Compare two metric values (return True if value1 is better than value2)."""
        return value1 > value2


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    type: PluginType
    cls: Type[Plugin]
    version: str = "1.0.0"
    description: str = ""
    enabled: bool = True
    priority: int = 0
    tags: List[str] = field(default_factory=list)


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a plugin is not found."""
    pass


class PluginLoadingError(PluginError):
    """Raised when a plugin fails to load."""
    pass


class PluginManager:
    """Manages the loading and retrieval of plugins."""
    
    _instance: Optional['PluginManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._plugins: Dict[str, PluginMetadata] = {}
        self._plugin_dirs: List[str] = []
        self._default_plugin_dir = os.path.join(
            os.path.dirname(__file__), "builtin_plugins"
        )
        self._initialized = True
    
    def add_plugin_dir(self, path: str) -> None:
        """Add a directory to search for plugins."""
        if os.path.isdir(path) and path not in self._plugin_dirs:
            self._plugin_dirs.append(path)
            # Add to Python path if not already there
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def load_plugin_from_file(self, file_path: str) -> None:
        """Load a plugin from a Python file."""
        if not os.path.exists(file_path):
            raise PluginNotFoundError(f"Plugin file not found: {file_path}")
        
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise PluginLoadingError(f"Could not load plugin from: {file_path}")
        
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise PluginLoadingError(f"Error loading plugin {file_path}: {e}") from e
        
        self._load_plugins_from_module(module)
    
    def load_plugins_from_module(self, module_name: str) -> None:
        """Load plugins from a Python module."""
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise PluginLoadingError(f"Could not import module {module_name}: {e}") from e
        
        self._load_plugins_from_module(module)
    
    def _load_plugins_from_module(self, module) -> None:
        """Load all plugins from a module."""
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Plugin)
                and obj != Plugin
                and not inspect.isabstract(obj)
            ):
                self._register_plugin(obj)
    
    def _register_plugin(self, plugin_cls: Type[Plugin]) -> None:
        """Register a plugin class."""
        name = plugin_cls.get_name()
        plugin_type = plugin_cls.get_type()
        description = plugin_cls.get_description()
        version = plugin_cls.get_version()
        
        if name in self._plugins:
            # Update existing plugin if it has lower priority
            existing = self._plugins[name]
            if existing.cls == plugin_cls:
                return  # Already registered
            # Keep the one with higher priority
            if existing.priority >= getattr(plugin_cls, "priority", 0):
                return
        
        metadata = PluginMetadata(
            name=name,
            type=plugin_type,
            cls=plugin_cls,
            version=version,
            description=description,
            priority=getattr(plugin_cls, "priority", 0),
            tags=getattr(plugin_cls, "tags", []),
        )
        self._plugins[name] = metadata
    
    def load_plugins(self) -> None:
        """Load all plugins from the configured directories."""
        # Load built-in plugins first
        if os.path.exists(self._default_plugin_dir):
            self._load_plugins_from_dir(self._default_plugin_dir)
        
        # Load plugins from additional directories
        for plugin_dir in self._plugin_dirs:
            self._load_plugins_from_dir(plugin_dir)
    
    def _load_plugins_from_dir(self, directory: str) -> None:
        """Load all plugins from a directory."""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    file_path = os.path.join(root, file)
                    try:
                        self.load_plugin_from_file(file_path)
                    except PluginError:
                        # Skip problematic plugins
                        continue
    
    def get_plugin(self, name: str) -> Type[Plugin]:
        """Get a plugin by name."""
        if name not in self._plugins:
            raise PluginNotFoundError(f"Plugin not found: {name}")
        return self._plugins[name].cls
    
    def get_plugins(self, plugin_type: Optional[PluginType] = None) -> List[Type[Plugin]]:
        """Get all plugins, optionally filtered by type."""
        plugins = []
        for metadata in self._plugins.values():
            if not metadata.enabled:
                continue
            if plugin_type is None or metadata.type == plugin_type:
                plugins.append(metadata.cls)
        return plugins
    
    def get_plugin_metadata(self, name: str) -> PluginMetadata:
        """Get metadata for a plugin."""
        if name not in self._plugins:
            raise PluginNotFoundError(f"Plugin not found: {name}")
        return self._plugins[name]
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """List all plugins with their metadata."""
        plugins = []
        for metadata in self._plugins.values():
            if plugin_type is not None and metadata.type != plugin_type:
                continue
            plugins.append({
                "name": metadata.name,
                "type": metadata.type.name,
                "version": metadata.version,
                "description": metadata.description,
                "enabled": metadata.enabled,
                "tags": metadata.tags,
            })
        return plugins
    
    def enable_plugin(self, name: str) -> None:
        """Enable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = True
    
    def disable_plugin(self, name: str) -> None:
        """Disable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = False
    
    def clear(self) -> None:
        """Clear all loaded plugins."""
        self._plugins.clear()


# Global plugin manager instance
plugin_manager = PluginManager()


def load_plugins() -> None:
    """Load all plugins using the global plugin manager."""
    plugin_manager.load_plugins()


def get_plugin(name: str) -> Type[Plugin]:
    """Get a plugin by name using the global plugin manager."""
    return plugin_manager.get_plugin(name)


def get_plugins(plugin_type: Optional[PluginType] = None) -> List[Type[Plugin]]:
    """Get all plugins, optionally filtered by type, using the global plugin manager."""
    return plugin_manager.get_plugins(plugin_type)


def register_plugin(plugin_cls: Type[Plugin]) -> Type[Plugin]:
    """Decorator to register a plugin class."""
    plugin_manager._register_plugin(plugin_cls)
    return plugin_cls
