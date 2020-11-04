import glob
import os
import pickle
import shutil
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

import lockfile

import numpy as np

from sklearn.pipeline import Pipeline

from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.util import logging_ as logging


__all__ = [
    'Backend'
]


def create(
    temporary_directory: str,
    output_directory: Optional[str],
    delete_tmp_folder_after_terminate: bool = True,
    delete_output_folder_after_terminate: bool = True,
) -> 'Backend':
    context = BackendContext(temporary_directory, output_directory,
                             delete_tmp_folder_after_terminate,
                             delete_output_folder_after_terminate,
                             )
    backend = Backend(context)

    return backend


def get_randomized_directory_name(temporary_directory: Optional[str] = None) -> str:
    uuid_str = str(uuid.uuid1(clock_seq=os.getpid()))

    temporary_directory = (
        temporary_directory
        if temporary_directory
        else os.path.join(
            tempfile.gettempdir(),
            "autosklearn_tmp_{}".format(
                uuid_str,
            ),
        )
    )

    return temporary_directory


class BackendContext(object):

    def __init__(self,
                 temporary_directory: str,
                 output_directory: Optional[str],
                 delete_tmp_folder_after_terminate: bool,
                 delete_output_folder_after_terminate: bool,
                 ):

        # Check that the names of tmp_dir and output_dir is not the same.
        if temporary_directory == output_directory and temporary_directory is not None:
            raise ValueError("The temporary and the output directory "
                             "must be different.")

        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        # attributes to check that directories were created by autosklearn.
        self._tmp_dir_created = False
        self._output_dir_created = False

        self._temporary_directory = (
            get_randomized_directory_name(
                temporary_directory=temporary_directory,
            )
        )
        self._output_directory = output_directory
        self._logger = logging.get_logger(__name__)
        self.create_directories()

    @property
    def output_directory(self) -> Optional[str]:
        if self._output_directory is not None:
            # make sure that tilde does not appear on the path.
            return os.path.expanduser(os.path.expandvars(self._output_directory))
        else:
            return None

    @property
    def temporary_directory(self) -> str:
        # make sure that tilde does not appear on the path.
        return os.path.expanduser(os.path.expandvars(self._temporary_directory))

    def create_directories(self) -> None:
        # Exception is raised if self.temporary_directory already exists.
        os.makedirs(self.temporary_directory)
        self._tmp_dir_created = True

        # Exception is raised if self.output_directory already exists.
        if self.output_directory is not None:
            os.makedirs(self.output_directory)
            self._output_dir_created = True

    def delete_directories(self, force: bool = True) -> None:
        if self.output_directory and (self.delete_output_folder_after_terminate or force):
            if self._output_dir_created is False:
                raise ValueError("Failed to delete output dir: %s because auto-sklearn did not "
                                 "create it. Please make sure that the specified output dir does "
                                 "not exist when instantiating auto-sklearn."
                                 % self.output_directory)
            try:
                shutil.rmtree(self.output_directory)
            except Exception:
                try:
                    if self._logger is not None:
                        self._logger.warning("Could not delete output dir: %s" %
                                             self.output_directory)
                    else:
                        print("Could not delete output dir: %s" % self.output_directory)
                except Exception:
                    print("Could not delete output dir: %s" % self.output_directory)

        if self.delete_tmp_folder_after_terminate or force:
            if self._tmp_dir_created is False:
                raise ValueError("Failed to delete tmp dir: % s because auto-sklearn did not "
                                 "create it. Please make sure that the specified tmp dir does not "
                                 "exist when instantiating auto-sklearn."
                                 % self.temporary_directory)
            try:
                shutil.rmtree(self.temporary_directory)
            except Exception:
                try:
                    if self._logger is not None:
                        self._logger.warning(
                            "Could not delete tmp dir: %s" % self.temporary_directory)
                    else:
                        print("Could not delete tmp dir: %s" % self.temporary_directory)
                except Exception:
                    print("Could not delete tmp dir: %s" % self.temporary_directory)


class Backend(object):
    """Utility class to load and save all objects to be persisted.

    These are:
    * start time of auto-sklearn
    * true targets of the ensemble
    """

    def __init__(self, context: BackendContext):
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
                raise ValueError("Output directory %s does not exist." % self.output_directory)

        self.internals_directory = os.path.join(self.temporary_directory, ".auto-sklearn")
        self._make_internals_directory()

    @property
    def output_directory(self) -> Optional[str]:
        return self.context.output_directory

    @property
    def temporary_directory(self) -> str:
        return self.context.temporary_directory

    def _make_internals_directory(self) -> None:
        try:
            os.makedirs(self.internals_directory)
        except Exception as e:
            self.logger.debug("_make_internals_directory: %s" % e)
        try:
            os.makedirs(self.get_runs_directory())
        except Exception as e:
            self.logger.debug("_make_internals_directory: %s" % e)

    def _get_start_time_filename(self, seed: Union[str, int]) -> str:
        if isinstance(seed, str):
            seed = int(seed)
        return os.path.join(self.internals_directory, "start_time_%d" % seed)

    def save_start_time(self, seed: str) -> str:
        self._make_internals_directory()
        start_time = time.time()

        filepath = self._get_start_time_filename(seed)

        if not isinstance(start_time, float):
            raise ValueError("Start time must be a float, but is %s." % type(start_time))

        if os.path.exists(filepath):
            raise ValueError(
                "{filepath} already exist. Different seeds should be provided for different jobs."
            )

        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(filepath), delete=False) as fh:
            fh.write(str(start_time))
            tempname = fh.name
        os.rename(tempname, filepath)

        return filepath

    def load_start_time(self, seed: int) -> float:
        with open(self._get_start_time_filename(seed), 'r') as fh:
            start_time = float(fh.read())
        return start_time

    def get_smac_output_directory(self) -> str:
        return os.path.join(self.temporary_directory, 'smac3-output')

    def get_smac_output_directory_for_run(self, seed: int) -> str:
        return os.path.join(
            self.temporary_directory,
            'smac3-output',
            'run_%d' % seed
        )

    def _get_targets_ensemble_filename(self) -> str:
        return os.path.join(self.internals_directory,
                            "true_targets_ensemble.npy")

    def save_targets_ensemble(self, targets: np.ndarray) -> str:
        self._make_internals_directory()
        if not isinstance(targets, np.ndarray):
            raise ValueError('Targets must be of type np.ndarray, but is %s' %
                             type(targets))

        filepath = self._get_targets_ensemble_filename()

        # Try to open the file without locking it, this will reduce the
        # number of times where we erroneously keep a lock on the ensemble
        # targets file although the process already was killed
        try:
            existing_targets = np.load(filepath, allow_pickle=True)
            if existing_targets.shape[0] > targets.shape[0] or \
                    (existing_targets.shape == targets.shape and
                     np.allclose(existing_targets, targets)):

                return filepath
        except Exception:
            pass

        with lockfile.LockFile(filepath):
            if os.path.exists(filepath):
                with open(filepath, 'rb') as fh:
                    existing_targets = np.load(fh, allow_pickle=True)
                    if existing_targets.shape[0] > targets.shape[0] or \
                            (existing_targets.shape == targets.shape and
                             np.allclose(existing_targets, targets)):
                        return filepath

            with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                    filepath), delete=False) as fh_w:
                np.save(fh_w, targets.astype(np.float32))
                tempname = fh_w.name

            os.rename(tempname, filepath)

        return filepath

    def load_targets_ensemble(self) -> np.ndarray:
        filepath = self._get_targets_ensemble_filename()

        with lockfile.LockFile(filepath):
            with open(filepath, 'rb') as fh:
                targets = np.load(fh, allow_pickle=True)

        return targets

    def _get_datamanager_pickle_filename(self) -> str:
        return os.path.join(self.internals_directory, 'datamanager.pkl')

    def save_datamanager(self, datamanager: AbstractDataManager) -> str:
        self._make_internals_directory()
        filepath = self._get_datamanager_pickle_filename()

        with lockfile.LockFile(filepath):
            if not os.path.exists(filepath):
                with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                        filepath), delete=False) as fh:
                    pickle.dump(datamanager, fh, -1)
                    tempname = fh.name
                os.rename(tempname, filepath)

        return filepath

    def load_datamanager(self) -> AbstractDataManager:
        filepath = self._get_datamanager_pickle_filename()
        with lockfile.LockFile(filepath):
            with open(filepath, 'rb') as fh:
                return pickle.load(fh)

    def get_runs_directory(self) -> str:
        return os.path.join(self.internals_directory, 'runs')

    def get_numrun_directory(self, seed: int, num_run: int, budget: float) -> str:
        return os.path.join(self.internals_directory, 'runs', '%d_%d_%s' % (seed, num_run, budget))

    def get_model_filename(self, seed: int, idx: int, budget: float) -> str:
        return '%s.%s.%s.model' % (seed, idx, budget)

    def get_cv_model_filename(self, seed: int, idx: int, budget: float) -> str:
        return '%s.%s.%s.cv_model' % (seed, idx, budget)

    def list_all_models(self, seed: int) -> List[str]:
        runs_directory = self.get_runs_directory()
        model_files = glob.glob(
            os.path.join(glob.escape(runs_directory), '%d_*' % seed, '%s.*.*.model' % seed)
        )
        return model_files

    def load_models_by_identifiers(self, identifiers: List[Tuple[int, int, float]]
                                   ) -> Dict:
        models = dict()

        for identifier in identifiers:
            seed, idx, budget = identifier
            models[identifier] = self.load_model_by_seed_and_id_and_budget(
                seed, idx, budget)

        return models

    def load_model_by_seed_and_id_and_budget(self, seed: int,
                                             idx: int,
                                             budget: float
                                             ) -> Pipeline:
        model_directory = self.get_numrun_directory(seed, idx, budget)

        model_file_name = '%s.%s.%s.model' % (seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, 'rb') as fh:
            return pickle.load(fh)

    def load_cv_models_by_identifiers(self, identifiers: List[Tuple[int, int, float]]
                                      ) -> Dict:
        models = dict()

        for identifier in identifiers:
            seed, idx, budget = identifier
            models[identifier] = self.load_cv_model_by_seed_and_id_and_budget(
                seed, idx, budget)

        return models

    def load_cv_model_by_seed_and_id_and_budget(self,
                                                seed: int,
                                                idx: int,
                                                budget: float
                                                ) -> Pipeline:
        model_directory = self.get_numrun_directory(seed, idx, budget)

        model_file_name = '%s.%s.%s.cv_model' % (seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, 'rb') as fh:
            return pickle.load(fh)

    def save_numrun_to_dir(
        self, seed: int, idx: int, budget: float, model: Optional[Pipeline],
        cv_model: Optional[Pipeline], ensemble_predictions: Optional[np.ndarray],
        valid_predictions: Optional[np.ndarray], test_predictions: Optional[np.ndarray],
    ) -> None:
        runs_directory = self.get_runs_directory()
        tmpdir = tempfile.mkdtemp(dir=runs_directory)
        if model is not None:
            file_path = os.path.join(tmpdir, self.get_model_filename(seed, idx, budget))
            with open(file_path, 'wb') as fh:
                pickle.dump(model, fh, -1)

        if cv_model is not None:
            file_path = os.path.join(tmpdir, self.get_cv_model_filename(seed, idx, budget))
            with open(file_path, 'wb') as fh:
                pickle.dump(cv_model, fh, -1)

        for preds, subset in (
            (ensemble_predictions, 'ensemble'),
            (valid_predictions, 'valid'),
            (test_predictions, 'test')
        ):
            if preds is not None:
                file_path = os.path.join(
                    tmpdir,
                    self.get_prediction_filename(subset, seed, idx, budget)
                )
                with open(file_path, 'wb') as fh:
                    pickle.dump(preds.astype(np.float32), fh, -1)
        try:
            os.rename(tmpdir, self.get_numrun_directory(seed, idx, budget))
        except OSError:
            if os.path.exists(self.get_numrun_directory(seed, idx, budget)):
                os.rename(self.get_numrun_directory(seed, idx, budget),
                          os.path.join(runs_directory, tmpdir + '.old'))
                os.rename(tmpdir, self.get_numrun_directory(seed, idx, budget))
                shutil.rmtree(os.path.join(runs_directory, tmpdir + '.old'))

    def get_ensemble_dir(self) -> str:
        return os.path.join(self.internals_directory, 'ensembles')

    def load_ensemble(self, seed: int) -> Optional[AbstractEnsemble]:
        ensemble_dir = self.get_ensemble_dir()

        if not os.path.exists(ensemble_dir):
            self.logger.warning('Directory %s does not exist' % ensemble_dir)
            return None

        if seed >= 0:
            indices_files = glob.glob(
                os.path.join(glob.escape(ensemble_dir), '%s.*.ensemble' % seed)
            )
            indices_files.sort()
        else:
            indices_files = os.listdir(ensemble_dir)
            indices_files = [os.path.join(ensemble_dir, f) for f in indices_files]
            indices_files.sort(key=lambda f: time.ctime(os.path.getmtime(f)))

        with open(indices_files[-1], 'rb') as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        return ensemble_members_run_numbers

    def save_ensemble(self, ensemble: AbstractEnsemble, idx: int, seed: int) -> None:
        try:
            os.makedirs(self.get_ensemble_dir())
        except Exception:
            pass

        filepath = os.path.join(
            self.get_ensemble_dir(),
            '%s.%s.ensemble' % (str(seed), str(idx).zfill(10))
        )
        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(ensemble, fh)
            tempname = fh.name
        os.rename(tempname, filepath)

    def get_prediction_filename(self, subset: str,
                                automl_seed: Union[str, int],
                                idx: int,
                                budget: float
                                ) -> str:
        return 'predictions_%s_%s_%s_%s.npy' % (subset, automl_seed, idx, budget)

    def save_predictions_as_txt(self,
                                predictions: np.ndarray,
                                subset: str,
                                idx: int, precision: int,
                                prefix: Optional[str] = None) -> None:
        if not self.output_directory:
            return
        # Write prediction scores in prescribed format
        filepath = os.path.join(
            self.output_directory,
            ('%s_' % prefix if prefix else '') + '%s_%s.predict' % (subset, str(idx)),
        )

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

    def write_txt_file(self, filepath: str, data: str, name: str) -> None:
        with lockfile.LockFile(filepath):
            with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                    filepath), delete=False) as fh:
                fh.write(data)
                tempname = fh.name
            os.rename(tempname, filepath)
            self.logger.debug('Created %s file %s' % (name, filepath))
