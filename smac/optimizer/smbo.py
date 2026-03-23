"""Mock smac.optimizer.smbo module"""
import time
import logging

class SMBO:
    def __init__(self, scenario, stats, runhistory, runhistory2epm,
                 initial_design, intensifier, aggregate_func,
                 model, acq_optimizer, acquisition_func, random_configuration_chooser,
                 max_run_budget=1, min_improvement=0.0, **kwargs):
        self.scenario = scenario
        self.stats = stats
        self.runhistory = runhistory
        self.runhistory2epm = runhistory2epm
        self.initial_design = initial_design
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.random_configuration_chooser = random_configuration_chooser
        self.max_run_budget = max_run_budget
        self.min_improvement = min_improvement
        
        self.logger = logging.getLogger("SMBO")
        self.start_time = None
        self.incumbent = None
        self.continue_optimization = True
        
    def start(self):
        self.start_time = time.time()
        
    def run(self):
        pass
        
    def shutdown(self):
        pass
        
    def optimize(self):
        return self.incumbent
        
    def get_next_configurations(self, num_configurations):
        return []
        
    def add_observations(self, observations):
        pass
        
    def choose_next(self, num_configurations=1):
        return []
