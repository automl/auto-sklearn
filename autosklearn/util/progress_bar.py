from typing import Any

import datetime
import time
from threading import Thread

from tqdm import trange


class ProgressBar(Thread):
    """A Thread that displays a tqdm progress bar in the console.

    It is specialized to display information relevant to fitting to the training data
    with auto-sklearn.

    Parameters
    ----------
    total : int
        The total amount that should be reached by the progress bar once it finishes
    update_interval : float
        Specifies how frequently the progress bar is updated (in seconds)
    disable : bool
        Turns on or off the progress bar. If True, this thread won't be started or
        initialized.
    kwargs : Any
        Keyword arguments that are passed into tqdm's constructor. Refer to:
        `tqdm <https://tqdm.github.io/docs/tqdm/>`_. Note that postfix can not be
        specified in the kwargs since it is already passed into tqdm by this class.
    """

    def __init__(
        self,
        total: int,
        update_interval: float = 1.0,
        disable: bool = False,
        **kwargs: Any,
    ):
        self.disable = disable
        if not disable:
            super().__init__(name="_progressbar_")
            self.total = total
            self.update_interval = update_interval
            self.terminated: bool = False
            self.kwargs = kwargs
            # start this thread
            self.start()

    def run(self) -> None:
        """Display a tqdm progress bar in the console.

        Additionally, it shows useful information related to the task. This method
        overrides the run method of Thread.
        """
        if not self.disable:
            for _ in trange(
                self.total,
                postfix=f"The total time budget for this task is "
                f"{datetime.timedelta(seconds=self.total)}",
                **self.kwargs,
            ):
                if not self.terminated:
                    time.sleep(self.update_interval)

    def stop(self) -> None:
        """Terminates the thread."""
        if not self.disable:
            self.terminated = True
            super().join()
