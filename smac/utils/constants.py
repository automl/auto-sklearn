"""Mock smac.utils.constants module"""
import numpy as np

MAXINT = np.iinfo(np.int32).max
MAXFLOAT = np.finfo(np.float64).max
MINFLOAT = np.finfo(np.float64).min

# Constant for random search
RANDOMSEARCH = 'random_search'

# Constant for multi-objective support
MULTI_OBJECTIVE_ALGORITHMS = ['EPSNET']
