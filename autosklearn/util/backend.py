from __future__ import print_function
import glob
import os
import tempfile
import time
import random
import lockfile
import numpy as np
import shutil
import six.moves.cPickle as pickle
from autosklearn.util import logging_ as logging


__all__ = [
    'Backend'
]


def create(temporary_directory,
           output_directory,
           delete_tmp_folder_after_terminate=True,
           delete_output_folder_after_terminate=True):
    context = BackendContext(temporary_directory, output_directory,
                             delete_tmp_folder_after_terminate,
                             delete_output_folder_after_terminate)
    backend = Backend(context)

    return backend


class BackendContext(object):

    def __init__(self,
                 temporary_directory,
                 output_directory,
                 delete_tmp_folder_after_terminate,
                 delete_output_folder_after_terminate):
        self._prepare_directories(temporary_directory, output_directory)
        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        self._logger = logging.get_logger(__name__)
        self.create_directories()

    @property
    def output_directory(self):
        return self.__output_directory

    @property
    def temporary_directory(self):
        return self.__temporary_directory

    def _prepare_directories(self, temporary_directory, output_directory):
        random_number = random.randint(0, 10000)
        pid = os.getpid()

        self.__temporary_directory = temporary_directory \
            if temporary_directory \
            else '/tmp/autosklearn_tmp_%d_%d' % (pid, random_number)

        self.__output_directory = output_directory \
            if output_directory \
            else '/tmp/autosklearn_output_%d_%d' % (pid, random_number)

    def create_directories(self):
        try:
            os.makedirs(self.temporary_directory)
        except OSError:
            pass
        try:
            os.makedirs(self.output_directory)
        except OSError:
            pass

    def __del__(self):
        self.delete_directories(force=False)

    def delete_directories(self, force=True):
        if self.delete_output_folder_after_terminate or force:
            try:
                shutil.rmtree(self.output_directory)
            except Exception:
                if self._logger is not None:
                    self._logger.warning("Could not delete output dir: %s" %
                                         self.output_directory)
                else:
                    print("Could not delete output dir: %s" %
                          self.output_directory)

        if self.delete_tmp_folder_after_terminate or force:
            try:
                shutil.rmtree(self.temporary_directory)
            except Exception:
                if self._logger is not None:
                    self._logger.warning("Could not delete tmp dir: %s" %
                                  self.temporary_directory)
                    pass
                else:
                    print("Could not delete tmp dir: %s" %
                          self.temporary_directory)


class Backend(object):
    """Utility class to load and save all objects to be persisted.

    These are:
    * start time of auto-sklearn
    * true targets of the ensemble
    """

    def __init__(self, context):
        self.logger = logging.get_logger(__name__)
        self.context = context

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

    @property
    def output_directory(self):
        return self.context.output_directory

    @property
    def temporary_directory(self):
        return self.context.temporary_directory

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

        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(filepath),
                delete=False) as fh:
            fh.write(str(start_time))
            tempname = fh.name
        os.rename(tempname, filepath)

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

            with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                    filepath), delete=False) as fh:
                np.save(fh, targets.astype(np.float32))
                tempname = fh.name

            os.rename(tempname, filepath)

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
                with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                        filepath), delete=False) as fh:
                    pickle.dump(datamanager, fh, -1)
                    tempname = fh.name
                os.rename(tempname, filepath)

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

        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(model, fh, -1)
            tempname = fh.name
        os.rename(tempname, filepath)

    def load_all_models(self, seed):
        model_directory = self.get_model_dir()

        if seed >= 0:
            model_files = glob.glob(os.path.join(model_directory,
                                                 '%s.*.model' % seed))
        else:
            model_files = os.listdir(model_directory)
            model_files = [os.path.join(model_directory, mf) for mf in model_files]

        models = self.load_models_by_file_names(model_files)

        return models

    def load_models_by_file_names(self, model_file_names):
        models = dict()

        for model_file in model_file_names:
            # File names are like: {seed}.{index}.model
            if model_file.endswith('/'):
                model_file = model_file[:-1]
            if not model_file.endswith('.model'):
                continue

            basename = os.path.basename(model_file)

            basename_parts = basename.split('.')
            seed = int(basename_parts[0])
            idx = int(basename_parts[1])

            models[(seed, idx)] = self.load_model_by_seed_and_id(seed, idx)

        return models

    def load_models_by_identifiers(self, identifiers):
        models = dict()

        for identifier in identifiers:
            seed, idx = identifier
            models[identifier] = self.load_model_by_seed_and_id(seed, idx)

        return models

    def load_model_by_seed_and_id(self, seed, idx):
        model_directory = self.get_model_dir()
        model_file_name = '%s.%s.model' % (seed, idx)
        model_file_path = os.path.join(model_directory, model_file_name)

        with open(model_file_path, 'rb') as fh:
            return (pickle.load(fh))

    def get_ensemble_dir(self):
        return os.path.join(self.internals_directory, 'ensembles')

    def load_ensemble(self, seed):
        ensemble_dir = self.get_ensemble_dir()

        if not os.path.exists(ensemble_dir):
            self.logger.warning('Directory %s does not exist' % ensemble_dir)
            return None

        if seed >= 0:
            indices_files = glob.glob(os.path.join(ensemble_dir,
                                                   '%s.*.ensemble' % seed))
            indices_files.sort()
        else:
            indices_files = os.listdir(ensemble_dir)
            indices_files = [os.path.join(ensemble_dir, f) for f in indices_files]
            indices_files.sort(key=lambda f: time.ctime(os.path.getmtime(f)))

        with open(indices_files[-1], 'rb') as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        return ensemble_members_run_numbers

    def save_ensemble(self, ensemble, idx, seed):
        try:
            os.makedirs(self.get_ensemble_dir())
        except Exception:
            pass

        filepath = os.path.join(self.get_ensemble_dir(),
                                '%s.%s.ensemble' % (str(seed),
                                                    str(idx).zfill(10)))
        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(ensemble, fh)
            tempname = fh.name
        os.rename(tempname, filepath)

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

        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(predictions.astype(np.float32), fh, -1)
            tempname = fh.name
        os.rename(tempname, filepath)

    def save_predictions_as_txt(self, predictions, subset, idx, prefix=None,
                                precision=3):
        # Write prediction scores in prescribed format
        filepath = os.path.join(self.output_directory,
                                ('%s_' % prefix if prefix else '') +
                                 '%s_%s.predict' % (subset, str(idx).zfill(5)))

        format_string = '{:.%dg} ' % precision
        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                filepath), delete=False) as output_file:
            for row in predictions:
                if not isinstance(row, np.ndarray) and not isinstance(row, list):
                    row = [row]
                for val in row:
                    output_file.write(format_string.format(float(val)))
                output_file.write('\n')
            tempname = output_file.name
        os.rename(tempname, filepath)

    def write_txt_file(self, filepath, data, name):
        lock_file = filepath + '.lock'
        with lockfile.LockFile(lock_file):
            if not os.path.exists(lock_file):
                with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                        filepath), delete=False) as fh:
                    fh.write(data)
                    tempname = fh.name
                os.rename(tempname, filepath)
                self.logger.debug('Created %s file %s' % (name, filepath))
            else:
                self.logger.debug('%s file already present %s' %
                                  (name, filepath))
