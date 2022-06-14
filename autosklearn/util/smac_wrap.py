from __future__ import annotations

from typing import Callable, Union

from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

SMACCallback = Callable[[SMBO, RunInfo, RunValue, float], Union[bool, None]]


class SmacRunCallback(IncorporateRunResultCallback):
    def __init__(self, f: SMACCallback):
        self.f = f

    def __call__(
        self,
        smbo: SMBO,
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> bool | None:
        """
        Parameters
        ----------
        smbo: SMBO
            The SMAC SMBO object

        run_info: RunInfo
            Information about the run completed

        result: RunValue
            The results of the run

        time_left: float
            How much time is left for the remaining runs

        Returns
        -------
        bool | None
            If False is returned, the optimization loop will stop
        """
        return self.f(smbo, run_info, result, time_left)
