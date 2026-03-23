"""Mock smac.facade.smac_facade module"""
import logging
import numpy as np

class SMAC:
    def __init__(self, scenario, tae_runner=None, intensifier=None,
                 acquisition_function=None, model=None, runhistory=None,
                 runhistory2epm=None, initial_design=None, random_configuration_chooser=None,
                 acq_optimizer=None, initial_incumbent=None, stats=None, rng=None):
        self.scenario = scenario
        self.tae_runner = tae_runner
        self.intensifier = intensifier
        self.acquisition_function = acquisition_function
        self.model = model
        self.runhistory = runhistory
        self.runhistory2epm = runhistory2epm
        self.initial_design = initial_design
        self.random_configuration_chooser = random_configuration_chooser
        self.acq_optimizer = acq_optimizer
        self.initial_incumbent = initial_incumbent
        self.stats = stats
        self.rng = rng if rng is not None else np.random.RandomState(42)
        
        self.logger = logging.getLogger("SMAC")
        self.solver = None
        
    def optimize(self):
        return self.initial_incumbent
        
    def validate(self, config_mode='inc', instance_mode='train+test',
                 repetitions=1, n_jobs=-1, backend='threading'):
        return {}
        
    def get_optimizer(self):
        return self.solver
        
    def get_runhistory(self):
        return self.runhistory
        
    def get_trajectory(self):
        return []
        
    def get_incumbent(self):
        return self.initial_incumbent
        
    def merge_foreign_data(self, runhistory, other_runhistories):
        pass
