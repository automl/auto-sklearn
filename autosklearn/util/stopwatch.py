# -*- encoding: utf-8 -*-
"""Created on Dec 17, 2014.

@author: Katharina Eggensperger
@project: AutoML2015

"""
import sys
import time
from collections import OrderedDict
from typing import Tuple


class TimingTask(object):
    _cpu_tic = 0.0
    _cpu_tac = 0.0
    _cpu_dur = 0.0
    _wall_tic = 0.0
    _wall_tac = 0.0
    _wall_dur = 0.0

    def __init__(self, name: str):
        self._name = name
        self._cpu_tic = time.process_time()
        self._wall_tic = time.time()

    def stop(self) -> None:
        if not self._cpu_tac:
            self._cpu_tac = time.process_time()
            self._wall_tac = time.time()
            self._cpu_dur = self._cpu_tac - self._cpu_tic
            self._wall_dur = self._wall_tac - self._wall_tic
        else:
            sys.stdout.write('Task has already stopped\n')

    @property
    def name(self) -> str:
        return self._name

    @property
    def cpu_tic(self) -> float:
        return self._cpu_tic

    @property
    def cpu_tac(self) -> float:
        return self._cpu_tac

    @property
    def cpu_dur(self) -> float:
        return self._cpu_dur

    @property
    def wall_tic(self) -> float:
        return self._wall_tic

    @property
    def wall_tac(self) -> float:
        return self._wall_tac

    @property
    def wall_dur(self) -> float:
        return self._wall_dur

    @property
    def dur(self) -> Tuple[float, float]:
        return self._cpu_dur, self._wall_dur


class StopWatch:

    """Class to collect all timing tasks."""

    def __init__(self) -> None:
        self._tasks = OrderedDict()
        self._tasks['stopwatch_time'] = TimingTask('stopwatch_time')

    def insert_task(self, name: str, cpu_dur: float, wall_dur: float) -> None:
        if name not in self._tasks:
            self._tasks[name] = TimingTask(name)
            self._tasks[name].stop()
            self._tasks[name]._wall_dur = wall_dur
            self._tasks[name]._cpu_dur = cpu_dur

    def start_task(self, name: str) -> None:
        if name not in self._tasks:
            self._tasks[name] = TimingTask(name)

    def wall_elapsed(self, name: str) -> float:
        tmp = time.time()
        if name in self._tasks:
            if not self._tasks[name].wall_dur:
                tsk_start = self._tasks[name].wall_tic
                return tmp - tsk_start
            else:
                return self._tasks[name].wall_dur
        return 0.0

    def cpu_elapsed(self, name: str) -> float:
        tmp = time.process_time()
        if name in self._tasks:
            if not self._tasks[name].cpu_dur:
                tsk_start = self._tasks[name].cpu_tic
                return tmp - tsk_start
            else:
                return self._tasks[name].cpu_dur
        return 0.0

    def stop_task(self, name: str) -> None:
        try:
            self._tasks[name].stop()
        except KeyError:
            sys.stderr.write('There is no such task: %s\n' % name)

    def get_cpu_dur(self, name: str) -> float:
        try:
            return self._tasks[name].cpu_dur
        except KeyError:
            sys.stderr.write('There is no such task: %s\n' % name)
        return 0.0

    def get_wall_dur(self, name: str) -> float:
        try:
            return self._tasks[name].wall_dur
        except KeyError:
            sys.stderr.write('There is no such task: %s\n' % name)
        return 0.0

    def cpu_sum(self) -> float:
        """Return sum of CPU time for all so far finished tasks."""
        return sum([max(0, self._tasks[tsk].cpu_dur) for tsk in self._tasks])

    def wall_sum(self) -> float:
        """Return sum of CPU time for all so far finished tasks."""
        return sum([max(0, self._tasks[tsk].wall_dur) for tsk in self._tasks])

    def __repr__(self) -> str:
        ret_str = '| %10s | %10s | %10s | %10s | %10s | %10s | %10s |\n' % \
                  ('Name', 'CPUStart', 'CPUEnd', 'CPUDur', 'WallStart',
                   'WallEnd',
                   'WallDur')
        ret_str += '+' + '------------+' * 7 + '\n'
        offset = self._tasks['stopwatch_time'].wall_tic
        for tsk in self._tasks:
            if self._tasks[tsk].wall_tac:
                wall_tac = self._tasks[tsk].wall_tac - offset
            ret_str += '| %10s | %10.5f | %10.5f | %10.5f | %10s | %10s | %10s |\n' % \
                       (tsk, self._tasks[tsk].cpu_tic, self._tasks[tsk].cpu_tac,
                        self.cpu_elapsed(tsk),
                        self._tasks[tsk].wall_tic - offset,
                        wall_tac if self._tasks[tsk].wall_tac else False,
                        self.wall_elapsed(tsk))
        return ret_str
