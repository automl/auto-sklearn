import time
from threading import Thread

from tqdm import trange


class ProgressBar(Thread):
    """
    A Thread that displays a tqdm progress bar in the console.

    Parameters
    ----------
    total : float
        The total amount that should be reached by the progress bar once it finishes
    update_interval : float
        Specifies how frequently the progress bar is updated (in seconds)
    disable : bool
        Turns on or off the progress bar. If True, this thread won't be started or initialized
    final_message : str
        Optional message, which is printed out on a new line once the bar is maxed out.
    """

    def __init__(
        self,
        total: float,
        update_interval: float = 1.0,
        disable: bool = False,
        final_message: str = None,
    ):
        self.disable = disable
        if not disable:
            super().__init__(name="_progressbar_")
            self.total = total
            self.update_interval = update_interval
            self.final_message = final_message
            self.terminated: bool = False
            self.start()

    def run(self) -> None:
        """
        Overrides the run method of Thread. It displays a tqdm progress bar in the console.

        """
        if not self.disable:
            for _ in trange(
                self.total, colour="green", desc="Fitting to the training data"
            ):
                if not self.terminated:
                    time.sleep(self.update_interval)
            print(self.final_message)

    def stop(self) -> None:
        """
        Terminates the thread.
        """
        if not self.disable:
            self.terminated = True
            super().join()
