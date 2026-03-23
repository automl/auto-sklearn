"""Mock ConfigSpace.util module"""
import numpy as np

def deactivate_inactive_hyperparameters(configuration, configuration_space, vector=None):
    """Mock function to deactivate inactive hyperparameters"""
    return configuration

def fix_types(configuration, configuration_space):
    """Mock function to fix types"""
    return configuration

def get_one_exchange_neighborhood(configuration, configuration_space, number_of_steps=1):
    """Mock function to get neighborhood configurations"""
    return [configuration]

def generate_grid(configuration_space, num_steps_dict=None):
    """Mock function to generate grid"""
    return []

def get_random_neighbor(configuration, configuration_space, random=np.random):
    """Mock function to get random neighbor"""
    return configuration

def impute_default_values(configuration, configuration_space):
    """Mock function to impute default values"""
    return configuration

def get_conditional_hyperparameters(configuration_space):
    """Mock function to get conditional hyperparameters"""
    return {}
