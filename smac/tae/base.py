"""Mock smac.tae.base module"""
from enum import Enum

class StatusType(Enum):
    RUNNING = 1
    SUCCESS = 2
    TIMEOUT = 3
    MEMOUT = 4
    CRASHED = 5
    ABORTED = 6

class BaseRunner:
    def __init__(self, ta, stats=None, run_obj="runtime",
                 par_factor=1, cost_for_crash=float("inf"),
                 abort_on_first_run_crash=True):
        self.ta = ta
        self.stats = stats
        self.run_obj = run_obj
        self.par_factor = par_factor
        self.cost_for_crash = cost_for_crash
        self.abort_on_first_run_crash = abort_on_first_run_crash
        
    def run(self, config, instance=None, cutoff=None,
            seed=12345, instance_specific="0", capped=False, budget=0.0):
        pass
