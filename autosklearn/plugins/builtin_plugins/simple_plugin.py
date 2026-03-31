"""Simple test plugin that doesn't require external dependencies."""

from autosklearn.plugins import Plugin, PluginType, register_plugin


@register_plugin
class SimplePlugin(Plugin):
    """A simple test plugin."""
    
    @classmethod
    def get_name(cls) -> str:
        return "simple_plugin"
    
    @classmethod
    def get_type(cls) -> PluginType:
        return PluginType.ALGORITHM
    
    @classmethod
    def get_description(cls) -> str:
        return "A simple test plugin for verifying the plugin system."
    
    def execute(self, *args, **kwargs):
        """Execute the plugin."""
        return "Simple plugin executed successfully!"


@register_plugin
class AnotherSimplePlugin(Plugin):
    """Another simple test plugin."""
    
    @classmethod
    def get_name(cls) -> str:
        return "another_simple_plugin"
    
    @classmethod
    def get_type(cls) -> PluginType:
        return PluginType.METRIC
    
    @classmethod
    def get_description(cls) -> str:
        return "Another simple test plugin."
    
    def execute(self, *args, **kwargs):
        """Execute the plugin."""
        return "Another simple plugin executed!"
