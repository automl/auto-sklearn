import time

from threading import Thread
from tqdm import trange


class ProgressBar(Thread):
    """A Thread that displays a tqdm progress bar in the console."""

    def __init__(self, total: float, update_interval: float = 1.0, disable: bool = False):
        """
        Parameters
        ----------
        total: the total amount that the progress bar should reach
        update_interval: reduce this to update the progress bar more frequently
        disable: flag that turns on or off the progress bar. If false, then no thread is started or created.
        """
        self.disable = disable
        if not disable:
            super().__init__(name="_progressbar_")
            self.total = total
            self.update_interval = update_interval
            self.terminated: bool = False
            self.start()

    def run(self):
        if not self.disable:
            for _ in trange(self.total, colour="green"):
                if not self.terminated:
                    time.sleep(self.update_interval)
                else:
                    pass  # max out the bar

    def stop(self):
        if not self.disable:
            self.terminated = True
            super().join()

