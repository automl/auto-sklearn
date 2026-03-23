"""Mock smac.scenario.scenario module"""
import os

class Scenario:
    def __init__(self, scenario, **kwargs):
        self.scenario = scenario
        self.output_dir = kwargs.get('output_directory', '')
        self.paramfile = kwargs.get('paramfile', '')
        self.instance_file = kwargs.get('instance_file', '')
        self.train_inst_fn = kwargs.get('train_inst_fn', '')
        self.test_inst_fn = kwargs.get('test_inst_fn', '')
        self.feature_dict = kwargs.get('feature_dict', {})
        self.run_obj = kwargs.get('run_obj', 'runtime')
        self.overall_obj = kwargs.get('overall_obj', 'par10')
        self.cutoff = kwargs.get('cutoff', None)
        self.algo_runs_timelimit = kwargs.get('algo_runs_timelimit', None)
        self.wallclock_limit = kwargs.get('wallclock_limit', None)
        self.ta_run_limit = kwargs.get('ta_run_limit', None)
        self.memory_limit = kwargs.get('memory_limit', None)
        self.min_budget = kwargs.get('min_budget', None)
        self.max_budget = kwargs.get('max_budget', None)
        self.budget_subdivision = kwargs.get('budget_subdivision', None)
        self.budget_type = kwargs.get('budget_type', None)
        self.cs = kwargs.get('cs', None)
        self.cs_file = kwargs.get('cs_file', None)
        self.execdir = kwargs.get('execdir', os.getcwd())
        self.ta = kwargs.get('ta', None)
        self.n_workers = kwargs.get('n_workers', 1)
        self.initial_incumbent = kwargs.get('initial_incumbent', None)
        self.instance_specific = kwargs.get('instance_specific', '0')
        self.limit_resources = kwargs.get('limit_resources', True)
        self.log_level = kwargs.get('log_level', 'INFO')
        self.shared_model = kwargs.get('shared_model', False)
        self.abort_on_first_run_crash = kwargs.get('abort_on_first_run_crash', True)
        
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)
