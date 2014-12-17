'''
Created on Dec 17, 2014

@author: Katharina Eggensperger
@project: AutoML2015
'''
from collections import OrderedDict

import sys
import time
from AutoML2015.data.data_io import vprint



class TimingTask():
    _name = False
    _cpu_tic = False
    _cpu_tac = False
    _cpu_dur = False
    _wall_tic = False
    _wall_tac = False
    _wall_dur = False

    def __init__(self, name):
        self._name = name
        self._cpu_tic = time.clock()
        self._wall_tic = time.time()

    def stop(self):
        if not self._cpu_tac:
            self._cpu_tac = time.clock()
            self._wall_tac = time.time()
            self._cpu_dur = self._cpu_tac - self._cpu_tic
            self._wall_dur = self._wall_tac - self._wall_tic
        else:
            sys.stdout.write("Task has already stopped\n")

    @property
    def name(self):
        return self._name

    @property
    def cpu_tic(self):
        return self._cpu_tic

    @property
    def cpu_tac(self):
        return self._cpu_tac

    @property
    def cpu_dur(self):
        return self._cpu_dur

    @property
    def wall_tic(self):
        return self._wall_tic

    @property
    def wall_tac(self):
        return self._wall_tac

    @property
    def wall_dur(self):
        return self._wall_dur

    @property
    def dur(self):
        return self._cpu_dur, self._wall_dur


class StopWatch:
    """
    Class to collect all timing tasks
    """
    _tasks = None

    def __init__(self):
        self._tasks = OrderedDict()

    def start_task(self, name):
        if name not in self._tasks:
            self._tasks[name] = TimingTask(name)
        else:
            sys.stderr.write("You are already timing task: %s\n" % name)

    def stop_task(self, name):
        if name in self._tasks:
            self._tasks[name].stop()
        else:
            sys.stderr.write("There is no such task: %s\n" % name)

    def get_cpu_dur(self, name):
        if name in self._tasks:
            return self._tasks[name].cpu_dur
        else:
            sys.stderr.write("There is no such task: %s\n" % name)

    def get_wall_dur(self, name):
        if name in self._tasks:
            return self._tasks[name].wall_dur
        else:
            sys.stderr.write("There is no such task: %s\n" % name)

    def cpu_sum(self):
        """
        Return sum of CPU time for all so far finished tasks
        """
        return sum([max(0, self._tasks[tsk].cpu_dur) for tsk in self._tasks])

    def wall_sum(self):
        """
        Return sum of CPU time for all so far finished tasks
        """
        return sum([max(0, self._tasks[tsk].wall_dur) for tsk in self._tasks])

    def __repr__(self):
        ret_str = "| %10s | %10s | %10s | %10s | %10s | %10s | %10s |\n" % \
                  ("Name", "CPUStart", "CPUEnd", "CPUDur", "WallStart", "WallEnd",
                   "WallDur")
        ret_str += "+" + "------------+"*7 + "\n"
        for tsk in self._tasks:
            ret_str += "| %10s | %10.5f | %10.5f | %10.5f | %10s | %10s | %10s |\n" % \
                       (tsk, self._tasks[tsk].cpu_tic, self._tasks[tsk].cpu_tac,
                        self._tasks[tsk].cpu_dur,
                        self._tasks[tsk].wall_tic, self._tasks[tsk].wall_tac,
                        self._tasks[tsk].wall_dur)
        return ret_str

