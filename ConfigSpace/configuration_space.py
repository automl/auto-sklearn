"""Mock ConfigSpace.configuration_space module"""
from . import (
    ConfigurationSpace as _ConfigurationSpace,
    Configuration as _Configuration,
)

ConfigurationSpace = _ConfigurationSpace
Configuration = _Configuration
