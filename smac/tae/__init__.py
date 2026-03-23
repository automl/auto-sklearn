"""Mock smac.tae module"""
from enum import Enum

class StatusType(Enum):
    RUNNING = 1
    SUCCESS = 2
    TIMEOUT = 3
    MEMOUT = 4
    CRASHED = 5
    ABORTED = 6

class TAEAbortException(Exception):
    """Exception to indicate that the TAE should abort"""
    pass

class FirstRunCrashedException(TAEAbortException):
    """Exception to indicate that the first run crashed"""
    pass

class ExecuteTARun:
    def __init__(self, ta, stats, runhistory, run_obj, par_factor, cost_for_crash, abort_on_first_run_crash):
        self.ta = ta
        self.stats = stats
        self.runhistory = runhistory
        self.run_obj = run_obj
        self.par_factor = par_factor
        self.cost_for_crash = cost_for_crash
        self.abort_on_first_run_crash = abort_on_first_run_crash

class ExecuteTARunOld(ExecuteTARun):
    pass

class ExecuteTARunSimple(ExecuteTARun):
    pass

__all__ = [
    'StatusType',
    'TAEAbortException',
    'FirstRunCrashedException',
    'ExecuteTARun',
    'ExecuteTARunOld',
    'ExecuteTARunSimple',
]
