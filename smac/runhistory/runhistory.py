"""Mock runhistory module"""
from enum import Enum
import typing

class StatusType(Enum):
    SUCCESS = 1
    TIMEOUT = 2
    MEMOUT = 3
    CRASHED = 4
    ABORT = 5
    CENSORED = 6
    RUNNING = 7

RunInfo = typing.NamedTuple('RunInfo', [
    ('config', typing.Any),
    ('instance', typing.Optional[str]),
    ('instance_specific', typing.Optional[str]),
    ('seed', int),
    ('budget', typing.Optional[float]),
])

RunValue = typing.NamedTuple('RunValue', [
    ('cost', float),
    ('time', float),
    ('status', StatusType),
    ('additional_info', dict),
    ('starttime', float),
    ('endtime', float),
])

class RunHistory:
    def __init__(self):
        self.data = {}
    
    def add(self, config, cost, time, status, instance_id=None,
            seed=None, budget=None, starttime=0, endtime=0, additional_info=None):
        pass
    
    def get_cost(self, config):
        return 0.0
    
    def empty(self):
        return len(self.data) == 0
    
    def update_cost(self, config):
        pass
    
    def get_all_configs(self):
        return []
    
    def get_runs_for_config(self, config):
        return []
    
    def get_incumbent(self):
        return None
