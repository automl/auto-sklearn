import os
import time

import lockfile
import numpy as np


__all__ = [
    'Backend'
]


class Backend(object):
    """Utility class to load and save all objects to be persisted.

    These are:
    * start time of auto-sklearn
    * true targets of the ensemble
    """

    def __init__(self, output_directory, temporary_directory):
        self.output_directory = output_directory
        self.temporary_directory = temporary_directory

        # Create the temporary directory if it does not yet exist
        try:
            os.makedirs(self.temporary_directory)
        except Exception:
            pass
        # This does not have to exist or be specified
        if self.output_directory is not None:
            if not os.path.exists(self.output_directory):
                raise ValueError("Output directory %s does not exist." %
                                 self.output_directory)

        self.internals_directory = os.path.join(self.temporary_directory,
                                                ".auto-sklearn")

    def _make_internals_directory(self):
        try:
            os.makedirs(self.internals_directory)
        except Exception as e:
            print(e)
            pass

    def _get_start_time_filename(self):
        return os.path.join(self.internals_directory, "start_time.txt")

    def save_start_time(self):
        self._make_internals_directory()
        start_time = time.time()

        if not isinstance(start_time, float):
            raise ValueError("Start time must be a float, but is %s." %
                             type(start_time))

        with open(self._get_start_time_filename(), 'w') as fh:
            fh.write(str(start_time))

    def load_start_time(self):
        with open(self._get_start_time_filename(), 'r') as fh:
            start_time = float(fh.read())
        return start_time

    def _get_targets_ensemble_filename(self):
        return os.path.join(self.internals_directory,
                            "true_targets_ensemble.npy")

    def save_targets_ensemble(self, targets):
        self._make_internals_directory()
        if not isinstance(targets, np.ndarray):
            raise ValueError("Targets must be of type np.ndarray, but is %s" %
                             type(targets))

        filepath = self._get_targets_ensemble_filename()

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            if not os.path.exists(filepath):
                np.save(filepath, targets)

    def load_targets_ensemble(self):
        filepath = self._get_targets_ensemble_filename()

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            targets = np.load(filepath)

        return targets


