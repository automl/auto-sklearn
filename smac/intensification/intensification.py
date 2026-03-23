"""Mock smac.intensification.intensification module"""
import numpy as np

class Intensifier:
    def __init__(self, tae_runner, stats, traj_logger, instances,
                 instance_specifics=None, run_limit=1, maxR=2000,
                 minR=1, maxR_capping=np.inf, eps=0.05,
                 intensification_percentage=0.5, runhistory=None,
                 scenario=None, Rs=None, max_config_calls=2000):
        self.tae_runner = tae_runner
        self.stats = stats
        self.traj_logger = traj_logger
        self.instances = instances
        self.instance_specifics = instance_specifics or {}
        self.run_limit = run_limit
        self.maxR = maxR
        self.minR = minR
        self.maxR_capping = maxR_capping
        self.eps = eps
        self.intensification_percentage = intensification_percentage
        self.runhistory = runhistory
        self.scenario = scenario
        self.Rs = Rs
        self.max_config_calls = max_config_calls
        
    def intensify(self, challenger, incumbent, N,
                  incumbent_inst_feats=None, challenger_inst_feats=None):
        return challenger, N, 0, True

class AbstractAdaptiveCapableIntensifier(Intensifier):
    pass

class SuccessiveHalving(AbstractAdaptiveCapableIntensifier):
    def __init__(self, initial_budget=1, eta=3, **kwargs):
        super().__init__(**kwargs)
        self.initial_budget = initial_budget
        self.eta = eta
        
class Hyperband(AbstractAdaptiveCapableIntensifier):
    def __init__(self, initial_budget=1, eta=3, max_iter=10, **kwargs):
        super().__init__(**kwargs)
        self.initial_budget = initial_budget
        self.eta = eta
        self.max_iter = max_iter
