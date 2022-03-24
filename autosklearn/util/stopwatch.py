from __future__ import annotations

from typing import Iterator, Mapping, Optional, Tuple

import sys
import time
from contextlib import contextmanager
from itertools import repeat

from typing_extensions import Literal


class TimingTask:
    """A task to start"""

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            The name of the task
        """
        self.name = name
        self.cpu_start: Optional[float] = None
        self.cpu_end: Optional[float] = None
        self.wall_start: Optional[float] = None
        self.wall_end: Optional[float] = None

    def start(self) -> None:
        """Start the task"""
        if not self.started():
            self.cpu_start = time.process_time()
            self.wall_start = time.time()

    def stop(self) -> None:
        """Stop the task"""
        if not self.finished():
            self.cpu_end = time.process_time()
            self.wall_end = time.time()

    def finished(self) -> bool:
        """Whether this task has been finished"""
        return self.cpu_end is not None

    def started(self) -> bool:
        """Whether this task has been started"""
        return self.cpu_start is not None

    @property
    def cpu_duration(self) -> Optional[float]:
        """The duration according to cpu clock time"""
        if self.cpu_start and self.cpu_end:
            return self.cpu_end - self.cpu_start
        else:
            return None

    @property
    def wall_duration(self) -> Optional[float]:
        """The duration according to wall clock time"""
        if self.wall_end and self.wall_start:
            return self.wall_end - self.wall_start
        else:
            return None

    @property
    def durations(self) -> Tuple[Optional[float], Optional[float]]:
        """The durations (cpu, wall)"""
        return self.cpu_duration, self.wall_duration


class StopWatch(Mapping[str, TimingTask]):
    """Class to collect timing tasks."""

    def __init__(self) -> None:
        self.tasks = {"_stopwatch_": TimingTask("_stopwatch_")}
        self.tasks["_stopwatch_"].start()

    def __contains__(self, name: object) -> bool:
        return name in self.tasks

    def __getitem__(self, name: str) -> TimingTask:
        return self.tasks[name]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[str]:
        return iter(self.tasks)

    def start(self, name: str) -> None:
        """Start a given task with a name"""
        task = TimingTask(name)
        self.tasks[name] = task
        task.start()

    def stop(self, name: str) -> None:
        """Stop a given task"""
        if name in self.tasks:
            self.tasks[name].stop()
        else:
            sys.stderr.write(f"No task with name {name}")

    def total_cpu(self) -> float:
        """Return sum of CPU time for all so far finished tasks."""
        tasks = self.tasks.values()
        return sum([t.cpu_duration for t in tasks if t.cpu_duration is not None])

    def total_wall(self) -> float:
        """Return sum of wall clock time for all so far finished tasks."""
        tasks = self.tasks.values()
        return sum([t.wall_duration for t in tasks if t.wall_duration is not None])

    def time_since(
        self,
        name: str,
        phase: Literal["start", "end"] = "start",
        default: float = 0.0,
        raises: bool = False,
    ) -> float:
        """The wall clock time since a task either began or ended

        Parameters
        ----------
        name : str
            The name of the task

        phase : Literal["start", "end"] = "start"
            From which phase you want to know the time elapsed since

        default: float = 0.0
            If None (default) then an error is raised if an answer can't be given.

        raises: bool = False
            Whether the method should raise if it can't find a valid time

        Returns
        -------
        float
            The time elapsed

        Raises
        ------
        ValueError
            If no default is specified and
            * the task has not been registered
            * the "start" and the task never started
            * the "end" and the task never started
        """
        task = self.tasks.get(name, None)

        if task is None:
            if raises:
                raise ValueError(f"Task not listed in {list(self.tasks.keys())}")
            else:
                return default

        if phase == "start":
            event_time = task.wall_start
        elif phase == "end":
            event_time = task.wall_end
        else:
            raise NotImplementedError()

        if event_time is None:
            if raises:
                raise ValueError(f"Task {task} has no time for {phase}")
            else:
                return default

        return time.time() - event_time

    def wall_elapsed(self, name: str) -> float:
        """Get the currently elapsed wall time for a task"""
        if name not in self.tasks:
            return 0.0

        task = self.tasks[name]
        if task.wall_start is None:
            return 0.0

        if task.wall_duration is not None:
            return task.wall_duration
        else:
            return time.time() - task.wall_start

    def cpu_elapsed(self, name: str) -> float:
        """Get the currently elapsed cpu time for a task"""
        if name not in self.tasks:
            return 0.0

        task = self.tasks[name]
        if task.cpu_start is None:
            return 0.0

        if task.cpu_duration is not None:
            return task.cpu_duration
        else:
            return time.time() - task.cpu_start

    @contextmanager
    def time(self, name: str) -> Iterator[TimingTask]:
        """Start timing a task

        Parameters
        ----------
        name : str
            The name of the task to measure
        """
        task = TimingTask(name)
        self.tasks[name] = task
        task.start()
        yield task
        task.stop()

    def __str__(self) -> str:
        headers = [
            "Name",
            "CPUStart",
            "CPUEnd",
            "CPUDur",
            "WallStart",
            "WallEnd",
            "WallDur",
        ]
        header = "|".join([f"{h:10s}" for h in headers])

        sep = "-" * 12
        seperator = "+" + "+".join(repeat(sep, len(headers))) + "+"

        entries = []
        for name, task in self.tasks.items():
            entry = (
                f"{task.name:10s} | {task.cpu_start:10.5f} | {task.cpu_end: 10.5f} | "
                f"{task.cpu_duration:10.5f} | {task.wall_start: 10s} | "
                f"{task.wall_end:10s} | {task.wall_duration:10s} |"
            )
            entries.append(entry)

        return "\n".join([header, seperator] + entries)
