"""Mock smac.tae.dask_runner module"""
import time
import logging
import numpy as np

class DaskParallelRunner:
    def __init__(self,
                 runner,
                 client=None,
                 n_workers: int = None,
                 patience: int = 5,
                 sleep_interval: float = 0.25):
        self.runner = runner
        self.client = client
        self.n_workers = n_workers
        self.patience = patience
        self.sleep_interval = sleep_interval
        self.logger = logging.getLogger("DaskParallelRunner")
        
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
