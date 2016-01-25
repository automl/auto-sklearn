from __future__ import print_function
import glob
import os
import time

import lockfile
import numpy as np

import six.moves.cPickle as pickle

from autosklearn.util import logging_ as logging


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
        self.logger = logging.get_logger(__name__)

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
        self._make_internals_directory()

    def _make_internals_directory(self):
        try:
            os.makedirs(self.internals_directory)
        except Exception as e:
            self.logger.debug("_make_internals_directory: %s" % e)
            pass

    def _get_start_time_filename(self, seed):
        seed = int(seed)
        return os.path.join(self.internals_directory, "start_time_%d" % seed)

    def save_start_time(self, seed):
        self._make_internals_directory()
        start_time = time.time()

        filepath = self._get_start_time_filename(seed)

        if not isinstance(start_time, float):
            raise ValueError("Start time must be a float, but is %s." %
                             type(start_time))

        with open(filepath, 'w') as fh:
            fh.write(str(start_time))

        return filepath

    def load_start_time(self, seed):
        with open(self._get_start_time_filename(seed), 'r') as fh:
            start_time = float(fh.read())
        return start_time

    def _get_targets_ensemble_filename(self):
        return os.path.join(self.internals_directory,
                            "true_targets_ensemble.npy")

    def save_targets_ensemble(self, targets):
        self._make_internals_directory()
        if not isinstance(targets, np.ndarray):
            raise ValueError('Targets must be of type np.ndarray, but is %s' %
                             type(targets))

        filepath = self._get_targets_ensemble_filename()

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            if os.path.exists(filepath):
                existing_targets = np.load(filepath)
                if existing_targets.shape[0] > targets.shape[0] or \
                        (existing_targets.shape == targets.shape and
                         np.allclose(existing_targets, targets)):
                    return filepath

            np.save(filepath, targets.astype(np.float32))

        return filepath

    def load_targets_ensemble(self):
        filepath = self._get_targets_ensemble_filename()

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            targets = np.load(filepath)

        return targets

    def _get_datamanager_pickle_filename(self):
        return os.path.join(self.internals_directory, 'datamanager.pkl')

    def save_datamanager(self, datamanager):
        self._make_internals_directory()
        filepath = self._get_datamanager_pickle_filename()

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            if not os.path.exists(filepath):
                with open(filepath, 'wb') as fh:
                    pickle.dump(datamanager, fh, -1)

        return filepath

    def load_datamanager(self):
        filepath = self._get_datamanager_pickle_filename()
        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path, timeout=60):
            with open(filepath, 'rb') as fh:
                return pickle.load(fh)

    def get_model_dir(self):
        return os.path.join(self.internals_directory, 'models')

    def save_model(self, model, idx, seed):
        # This should fail if no models directory exists
        filepath = os.path.join(self.get_model_dir(),
                                '%s.%s.model' % (seed, idx))
        with open(filepath, 'wb') as fh:
            pickle.dump(model, fh, -1)

    def load_all_models(self, seed):
        model_directory = self.get_model_dir()

        if seed >= 0:
            model_files = glob.glob(os.path.join(model_directory,
                                                 '%s.*.model' % seed))
        else:
            model_files = os.listdir(model_directory)
            model_files = [os.path.join(model_directory, mf) for mf in model_files]

        models = dict()
        for model_file in model_files:
            # File names are like: {seed}.{index}.model
            if model_file.endswith('/'):
                model_file = model_file[:-1]
            basename = os.path.basename(model_file)
            automl_seed = int(basename.split('.')[0])
            idx = int(basename.split('.')[1])
            with open(os.path.join(model_directory, basename), 'rb') as fh:
                models[(automl_seed, idx)] = (pickle.load(fh))

        return models

    def get_ensemble_indices_dir(self):
        return os.path.join(self.internals_directory, 'ensemble_indices')

    def load_ensemble_indices_weights(self, seed):
        indices_dir = self.get_ensemble_indices_dir()

        if not os.path.exists(indices_dir):
            self.logger.warning('Directory %s does not exist' % indices_dir)
            return {}

        if seed >= 0:
            indices_files = glob.glob(os.path.join(indices_dir,
                                                   '%s.*.indices' % seed))
            indices_files.sort()
        else:
            indices_files = os.listdir(indices_dir)
            indices_files = [os.path.join(indices_dir, f) for f in indices_files]
            indices_files.sort(key=lambda f: time.ctime(os.path.getmtime(f)))

        with open(indices_files[-1], 'rb') as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        if len(ensemble_members_run_numbers) == 0:
            self.logger.error('Ensemble indices file %s does not contain any '
                              'ensemble information.', indices_files[-1])

        return ensemble_members_run_numbers

    def save_ensemble_indices_weights(self, indices, idx, seed):
        try:
            os.makedirs(self.get_ensemble_indices_dir())
        except Exception:
            pass

        filepath = os.path.join(self.get_ensemble_indices_dir(),
                                '%s.%s.indices' % (str(seed), str(idx).zfill(
                                    10)))
        with open(filepath, 'wb') as fh:
            pickle.dump(indices, fh)

    def _get_prediction_output_dir(self, subset):
        return os.path.join(self.internals_directory,
                            'predictions_%s' % subset)

    def save_predictions_as_npy(self, predictions, subset, automl_seed, idx):
        output_dir = self._get_prediction_output_dir(subset)
        # Make sure an output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, 'predictions_%s_%s_%s.npy' %
                                            (subset, automl_seed, str(idx)))

        with open(filepath, 'wb') as fh:
            pickle.dump(predictions.astype(np.float32), fh, -1)

    def save_predictions_as_txt(self, predictions, subset, idx, prefix=None):
        # Write prediction scores in prescribed format
        filepath = os.path.join(self.output_directory,
                                ('%s_' % prefix if prefix else '') +
                                 '%s_%s.predict' % (subset, str(idx).zfill(5)))

        with open(filepath, 'w') as output_file:
            for row in predictions:
                if not isinstance(row, np.ndarray) and not isinstance(row, list):
                    row = [row]
                for val in row:
                    output_file.write('{:g} '.format(float(val)))
                output_file.write('\n')

    def write_txt_file(self, filepath, data, name):
        lock_file = filepath + '.lock'
        with lockfile.LockFile(lock_file):
            if not os.path.exists(lock_file):
                with open(filepath, 'w') as fh:
                    fh.write(data)
                self.logger.debug('Created %s file %s' % (name, filepath))
            else:
                self.logger.debug('%s file already present %s' %
                                  (name, filepath))
