"""Mock smac.tae.serial_runner module"""
import numpy as np

class SerialRunner:
    def __init__(self,
                 runner,
                 n_worker: int = 1,
                 **kwargs):
        self.runner = runner
        self.n_worker = n_worker
        
    def submit(self,
               run,
               _cost_for_crash,
               **kwargs):
        pass
        
    def fetch_work(self,
                   run,
                   num: int = np.inf,
                   time_bound: int = np.inf,
                   _cost_for_crash = None):
        pass
        
    def run(self,
            config,
            instance=None,
            cutoff=None,
            seed=12345,
            instance_specific="0",
            capped=False,
            budget=0.0):
        pass
