#!/usr/bin/env python3
"""CLI for managing auto-sklearn plugins."""

import argparse
import sys
from typing import Optional

from autosklearn.plugins import (
    PluginManager,
    PluginType,
    plugin_manager,
    load_plugins as load_plugins_func,
)


def main():
    """Main entry point for the plugin CLI."""
    parser = argparse.ArgumentParser(
        description="auto-sklearn Plugin Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  auto-sklearn-plugins list
  auto-sklearn-plugins list --type algorithm
  auto-sklearn-plugins show custom_logistic_regression
  auto-sklearn-plugins load --dir ./my_plugins
  auto-sklearn-plugins enable custom_metric
  auto-sklearn-plugins disable old_algorithm
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all available plugins")
    list_parser.add_argument(
        "--type",
        choices=[t.name.lower() for t in PluginType],
        help="Filter by plugin type",
    )
    list_parser.add_argument(
        "--enabled", action="store_true", help="Show only enabled plugins"
    )
    list_parser.add_argument(
        "--disabled", action="store_true", help="Show only disabled plugins"
    )
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show details of a specific plugin")
    show_parser.add_argument("name", help="Name of the plugin")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load plugins from a directory")
    load_parser.add_argument(
        "--dir",
        action="append",
        dest="dirs",
        help="Directory to load plugins from (can be specified multiple times)",
    )
    load_parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Module to load plugins from (can be specified multiple times)",
    )
    
    # Enable command
    enable_parser = subparsers.add_parser("enable", help="Enable a plugin")
    enable_parser.add_argument("name", help="Name of the plugin to enable")
    
    # Disable command
    disable_parser = subparsers.add_parser("disable", help="Disable a plugin")
    disable_parser.add_argument("name", help="Name of the plugin to disable")
    
    # Reload command
    subparsers.add_parser("reload", help="Reload all plugins")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return handle_command(args)


def handle_command(args) -> int:
    """Handle the command."""
    try:
        if args.command == "list":
            return handle_list(args)
        elif args.command == "show":
            return handle_show(args)
        elif args.command == "load":
            return handle_load(args)
        elif args.command == "enable":
            return handle_enable(args)
        elif args.command == "disable":
            return handle_disable(args)
        elif args.command == "reload":
            return handle_reload(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_list(args) -> int:
    """Handle the list command."""
    plugin_type = None
    if args.type:
        try:
            plugin_type = PluginType[args.type.upper()]
        except KeyError:
            print(f"Invalid plugin type: {args.type}", file=sys.stderr)
            return 1
    
    # Load plugins if not already loaded
    if not plugin_manager._plugins:
        load_plugins_func()
    
    plugins = plugin_manager.list_plugins(plugin_type)
    
    # Filter by enabled/disabled
    if args.enabled:
        plugins = [p for p in plugins if p["enabled"]]
    elif args.disabled:
        plugins = [p for p in plugins if not p["enabled"]]
    
    if not plugins:
        print("No plugins found.")
        return 0
    
    # Print table header
    print(f"{'Name':<30} {'Type':<15} {'Version':<10} {'Enabled':<8} {'Description'}")
    print("-" * 100)
    
    # Print plugin info
    for plugin in plugins:
        enabled = "✓" if plugin["enabled"] else "✗"
        print(
            f"{plugin['name']:<30} "
            f"{plugin['type']:<15} "
            f"{plugin['version']:<10} "
            f"{enabled:<8} "
            f"{plugin['description'][:50]}"
        )
    
    return 0


def handle_show(args) -> int:
    """Handle the show command."""
    # Load plugins if not already loaded
    if not plugin_manager._plugins:
        load_plugins_func()
    
    try:
        metadata = plugin_manager.get_plugin_metadata(args.name)
    except Exception as e:
        print(f"Plugin not found: {args.name}", file=sys.stderr)
        return 1
    
    print(f"Plugin: {metadata.name}")
    print(f"Type: {metadata.type.name}")
    print(f"Version: {metadata.version}")
    print(f"Enabled: {'Yes' if metadata.enabled else 'No'}")
    print(f"Priority: {metadata.priority}")
    print(f"Tags: {', '.join(metadata.tags) if metadata.tags else 'None'}")
    print(f"\nDescription:\n{metadata.description}")
    print(f"\nClass: {metadata.cls.__module__}.{metadata.cls.__name__}")
    
    return 0


def handle_load(args) -> int:
    """Handle the load command."""
    # Load from directories
    if args.dirs:
        for directory in args.dirs:
            plugin_manager.add_plugin_dir(directory)
            print(f"Added plugin directory: {directory}")
    
    # Load from modules
    if args.modules:
        for module in args.modules:
            try:
                plugin_manager.load_plugins_from_module(module)
                print(f"Loaded plugins from module: {module}")
            except Exception as e:
                print(f"Failed to load module {module}: {e}", file=sys.stderr)
                return 1
    
    # Load all plugins
    load_plugins_func()
    
    plugin_count = len(plugin_manager._plugins)
    print(f"Loaded {plugin_count} plugin(s)")
    
    return 0


def handle_enable(args) -> int:
    """Handle the enable command."""
    plugin_manager.enable_plugin(args.name)
    print(f"Enabled plugin: {args.name}")
    return 0


def handle_disable(args) -> int:
    """Handle the disable command."""
    plugin_manager.disable_plugin(args.name)
    print(f"Disabled plugin: {args.name}")
    return 0


def handle_reload(args) -> int:
    """Handle the reload command."""
    plugin_manager.clear()
    load_plugins_func()
    
    plugin_count = len(plugin_manager._plugins)
    print(f"Reloaded {plugin_count} plugin(s)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
