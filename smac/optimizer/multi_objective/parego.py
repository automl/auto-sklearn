"""Mock smac.optimizer.multi_objective.parego module"""
import numpy as np

class ParEGO:
    def __init__(self,
                 scenario,
                 stats,
                 runhistory,
                 runhistory2epm,
                 aggregate_func,
                 model,
                 acq_optimizer,
                 acquisition_func,
                 max_run_budget=1,
                 min_improvement=0.0,
                 rng=None):
        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.runhistory2epm = runhistory2epm
        self.aggregate_func = aggregate_func
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.max_run_budget = max_run_budget
        self.min_improvement = min_improvement
        self.rng = rng if rng is not None else np.random.RandomState(42)
        
    def choose_next(self, num_configurations=1):
        return []
