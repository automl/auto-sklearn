"""Mock json reader/writer for ConfigSpace"""
import json

def read(json_string):
    """Read a configuration space from a JSON string"""
    from ConfigSpace import ConfigurationSpace
    return ConfigurationSpace()

def write(configuration_space):
    """Write a configuration space to a JSON string"""
    return json.dumps({})
