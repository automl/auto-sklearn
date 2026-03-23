"""Mock smac.runhistory.runhistory2epm module"""
import numpy as np
from typing import List, Dict, Any

class RunHistory2EPM:
    def __init__(self,
                 scenario,
                 num_params,
                 success_states=None,
                 impute_censored_data=False,
                 impute_state=None,
                 scale_perc=5,
                 min_samples_to_draw=10,
                 output_y_scaler=None,
                 seed=42):
        self.scenario = scenario
        self.num_params = num_params
        self.success_states = success_states
        self.impute_censored_data = impute_censored_data
        self.impute_state = impute_state
        self.scale_perc = scale_perc
        self.min_samples_to_draw = min_samples_to_draw
        self.output_y_scaler = output_y_scaler
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def build_matrix(self, runhistory, stats, num_samples=0):
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    def transform(self, runhistory, stats, num_samples=0):
        return self.build_matrix(runhistory, stats, num_samples)

class RunHistory2EPM4Cost(RunHistory2EPM):
    pass

class RunHistory2EPM4LogCost(RunHistory2EPM):
    pass

class RunHistory2EPM4EIPS(RunHistory2EPM):
    pass
