import datetime
import time
from threading import Thread

from tqdm import trange  # type: ignore


class ProgressBar(Thread):
    """
    A Thread that displays a tqdm progress bar in the console.

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
    """

    def __init__(
        self,
        total: int,
        update_interval: float = 1.0,
        disable: bool = False,
    ):
        self.disable = disable
        if not disable:
            super().__init__(name="_progressbar_")
            self.total = total
            self.update_interval = update_interval
            self.terminated: bool = False
            # start this thread
            self.start()

    def run(self) -> None:
        """
        Overrides the run method of Thread. It displays a tqdm progress bar in the
        console with useful descriptions about the task.

        """
        if not self.disable:
            for _ in trange(
                self.total,
                colour="green",
                desc="Fitting to the training data",
                postfix=f"The total time budget for this task is"
                f" {datetime.timedelta(seconds=self.total)}",
            ):
                if not self.terminated:
                    time.sleep(self.update_interval)
            print("Finishing up the task...")

    def stop(self) -> None:
        """
        Terminates the thread.
        """
        if not self.disable:
            self.terminated = True
            super().join()
