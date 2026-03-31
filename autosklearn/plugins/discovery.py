"""
Plugin discovery and loading utilities.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from autosklearn.plugins import PluginRegistry, PluginType


def discover_plugins_from_path(path: Path) -> Dict[str, List[str]]:
    """Discover plugins from a directory path.

    Args:
        path: Directory path to search for plugins

    Returns:
        Dictionary mapping plugin type to list of discovered plugin names
    """
    discovered: Dict[str, List[str]] = {pt.value: [] for pt in PluginType}
    registry = PluginRegistry.get_instance()

    if not path.exists():
        return discovered

    for py_file in path.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = str(py_file.with_suffix("")).replace("/", ".")

        try:
            module = importlib.import_module(module_name)
            for plugin_type in PluginType:
                for name in registry.list_plugins(plugin_type):
                    if name not in discovered[plugin_type.value]:
                        discovered[plugin_type.value].append(name)
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to load module {module_name}: {e}", RuntimeWarning)

    return discovered


def load_plugins_from_config(config_path: Path) -> None:
    """Load plugins specified in a configuration file.

    Args:
        config_path: Path to configuration file (YAML or JSON)
    """
    import json

    import yaml

    if not config_path.exists():
        return

    with open(config_path) as f:
        if config_path.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    registry = PluginRegistry.get_instance()

    plugins_config = config.get("plugins", {})
    for plugin_type_str, plugin_names in plugins_config.items():
        plugin_type = PluginType(plugin_type_str)
        for plugin_name in plugin_names:
            try:
                module_name, class_name = plugin_name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                plugin_class = getattr(module, class_name)
                registry.register_plugin(plugin_type, plugin_class)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load plugin {plugin_name}: {e}",
                    RuntimeWarning,
                )


def get_available_plugins() -> Dict[str, List[str]]:
    """Get all available plugins grouped by type.

    Returns:
        Dictionary mapping plugin type to list of plugin names
    """
    registry = PluginRegistry.get_instance()
    return {
        plugin_type.value: registry.list_plugins(plugin_type)
        for plugin_type in PluginType
    }


def validate_plugin_dependencies() -> Dict[str, Set[str]]:
    """Validate that all plugin dependencies are satisfied.

    Returns:
        Dictionary mapping plugin name to missing dependencies
    """
    registry = PluginRegistry.get_instance()
    missing: Dict[str, Set[str]] = {}

    for plugin_type in PluginType:
        for name, plugin_class in registry.get_all(plugin_type).items():
            metadata = plugin_class.get_metadata()
            for dep in metadata.requires:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    if name not in missing:
                        missing[name] = set()
                    missing[name].add(dep)

    return missing


__all__ = [
    "discover_plugins_from_path",
    "load_plugins_from_config",
    "get_available_plugins",
    "validate_plugin_dependencies",
]
