from __future__ import annotations

from typing import Any

import datetime
import time
from threading import Thread

from tqdm import trange


class ProgressBar(Thread):
    """A Thread that displays a tqdm progress bar in the console.

    Treat this class as an ordinary thread. So to display a progress bar,
    call start() on an instance of this class. To wait for the thread to
    terminate call join(), which will max out the progress bar,
    therefore terminate this thread immediately.

    Parameters
    ----------
    total : int
        The total amount that should be reached by the progress bar once it finishes.
    update_interval : float, default=1.0
        Specifies how frequently the progress bar is updated (in seconds).
    disable : bool, default=False
        Turns on or off the progress bar. If True, this thread does not get
        initialized and won't be started if start() is called.
    tqdm_kwargs : Any, optional
        Keyword arguments that are passed into tqdm's constructor. Refer to:
        `tqdm <https://tqdm.github.io/docs/tqdm/>`_ for a list of parameters that
        tqdm accepts. Note that 'postfix' cannot be specified in the kwargs since it is
        already passed into tqdm by this class.

    Examples
    --------

    .. code:: python

        progress_bar = ProgressBar(
            total=10,
            desc="Executing code that runs for 10 seconds",
            colour="green",
        )
        # colour is a tqdm parameter passed as a tqdm_kwargs
        try:
            progress_bar.start()
            # some code that runs for 10 seconds
        except SomeException:
            # something went wrong
        finally:
            progress_bar.join()
            # perform some cleanup
    """

    def __init__(
        self,
        total: int,
        update_interval: float = 1.0,
        disable: bool = False,
        **tqdm_kwargs: Any,
    ):
        self.disable = disable
        if not disable:
            super().__init__(name="_progressbar_")
            self.total = total
            self.update_interval = update_interval
            self.terminated: bool = False
            self.tqdm_kwargs = tqdm_kwargs

    def start(self) -> None:
        """Start a new thread that calls the run() method."""
        if not self.disable:
            super().start()

    def run(self) -> None:
        """Display a tqdm progress bar in the console."""
        if not self.disable:
            for _ in trange(
                self.total,
                postfix=f"The total time budget for this task is "
                f"{datetime.timedelta(seconds=self.total)}",
                **self.tqdm_kwargs,
            ):
                if not self.terminated:
                    time.sleep(self.update_interval)

    def join(self, timeout: float | None = None) -> None:
        """Maxes out the progress bar and thereby terminating this thread."""
        if not self.disable:
            self.terminated = True
            super().join(timeout)
