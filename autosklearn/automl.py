# -*- encoding: utf-8 -*-
import hashlib
import multiprocessing
import os
from os.path import join
import traceback

import numpy as np

import lockfile
from HPOlibConfigSpace.converters import pcs_parser
from sklearn.base import BaseEstimator

import six.moves.cPickle as pickle
from autosklearn import submit_process
from autosklearn.constants import *
from autosklearn.data import split_data
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.data.Xy_data_manager import XyDataManager
from autosklearn.metalearning import metalearning
from autosklearn.models import evaluator, paramsklearn
from autosklearn.util import StopWatch, get_logger


def _write_file_with_data(filepath, data, name, log_function):
    if _check_path_for_save(filepath, name, log_function):
        with open(filepath, 'w') as fh:
            fh.write(data)


def _check_path_for_save(filepath, name, log_function):
    result = False
    lock_file = filepath + '.lock'
    with lockfile.LockFile(lock_file):
        if not os.path.exists(lock_file):
            result = True
            log_function('Created %s file %s' % (name, filepath))
        else:
            log_function('%s file already present %s' % (name, filepath))
    return result


def _get_logger(log_dir, basename, seed):
    return get_logger(outputdir=log_dir,
                      name='AutoML_%s_%d' % (basename, seed))


def _set_auto_seed(seed):
    env_key = 'AUTOSKLEARN_SEED'
    # Set environment variable:
    env_seed = os.environ.get(env_key)
    if env_seed is not None and int(env_seed) != seed:
        raise ValueError('It seems you have already started an instance '
                         'of AutoSklearn in this thread.')
    else:
        os.environ[env_key] = str(seed)


def _get_auto_seed():
    value = os.environ['AUTOSKLEARN_SEED']
    assert value is not None
    return int(value)


def _del_auto_seed():
    env_key = 'AUTOSKLEARN_SEED'
    del os.environ[env_key]


def _run_smac(tmp_dir, basename, time_for_task, ml_memory_limit,
              data_manager_path, configspace_path, initial_configurations,
              per_run_time_limit, watcher, log_function):
    task_name = 'runSmac'
    watcher.start_task(task_name)

    # = Create an empty instance file
    instance_file = os.path.join(tmp_dir, 'instances.txt')
    _write_file_with_data(instance_file, 'holdout', 'Instances', log_function)

    # = Start SMAC
    time_smac = max(0, time_for_task - watcher.wall_elapsed(basename))
    log_function('Start SMAC with %5.2fsec time left' % time_smac)
    proc_smac, smac_call = \
        submit_process.run_smac(dataset_name=basename,
                                dataset=data_manager_path,
                                tmp_dir=tmp_dir,
                                searchspace=configspace_path,
                                instance_file=instance_file,
                                limit=time_smac,
                                cutoff_time=per_run_time_limit,
                                initial_challengers=initial_configurations,
                                memory_limit=ml_memory_limit,
                                seed=_get_auto_seed())
    log_function(smac_call)
    watcher.stop_task('runSmac')
    return proc_smac


def _run_ensemble_builder(tmp_dir,
                          output_dir,
                          basename,
                          time_for_task,
                          task,
                          metric,
                          ensemble_size,
                          ensemble_nbest,
                          ensemble_indices_dir,
                          watcher,
                          log_function):
    task_name = 'runEnsemble'
    watcher.start_task(task_name)
    time_left_for_ensembles = max(
        0, time_for_task -
           (watcher.wall_elapsed(basename)))
    log_function('Start Ensemble with %5.2fsec time left' %
                 time_left_for_ensembles)
    proc_ensembles = \
        submit_process.run_ensemble_builder(tmp_dir=tmp_dir,
                                            dataset_name=basename,
                                            task_type=task,
                                            metric=metric,
                                            limit=time_left_for_ensembles,
                                            output_dir=output_dir,
                                            ensemble_size=ensemble_size,
                                            ensemble_nbest=ensemble_nbest,
                                            seed=_get_auto_seed(),
                                            ensemble_indices_output_dir=ensemble_indices_dir)
    watcher.stop_task(task_name)
    return proc_ensembles


def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            metalearning_cnt, x_train, y_train, watcher,
                            log_function):
    # == Calculate metafeatures
    watcher.start_task('CalculateMetafeatures')
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    if metalearning_cnt <= 0:
        ml = None
    elif data_info_task in \
            [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION]:
        ml = metalearning.MetaLearning()
        log_function('Start calculating metafeatures for %s' % basename)
        ml.calculate_metafeatures_with_labels(x_train, y_train,
                                              categorical=categorical,
                                              dataset_name=basename)
    else:
        ml = None
        log_function('Metafeatures not calculated')
    watcher.stop_task('CalculateMetafeatures')
    log_function(
        'Calculating Metafeatures (categorical attributes) took %5.2f' %
        watcher.wall_elapsed('CalculateMetafeatures'))
    return ml


def _calculate_metafeatures_encoded(ml, basename, x_train, y_train, watcher,
                                    log_funciton):
    task_name = 'CalculateMetafeaturesEncoded'
    watcher.start_task(task_name)
    ml.calculate_metafeatures_encoded_labels(
        X_train=x_train,
        Y_train=y_train,
        categorical=[False] * x_train.shape[0],
        dataset_name=basename)
    watcher.stop_task(task_name)
    log_funciton(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec' %
        watcher.wall_elapsed(task_name))

def _create_search_space(tmp_dir, data_info, watcher, log_function):
    task_name = 'CreateConfigSpace'
    watcher.start_task(task_name)
    configspace_path = os.path.join(tmp_dir, 'space.pcs')
    configuration_space = paramsklearn.get_configuration_space(
        data_info)
    sp_string = pcs_parser.write(configuration_space)
    _write_file_with_data(configspace_path, sp_string,
                          'Configuration space', log_function)
    watcher.stop_task(task_name)

    return configuration_space, configspace_path


def _get_initial_configuration(ml, basename, metric, configuration_space,
                               task, metadata_directory,
                               initial_configurations_via_metalearning,
                               is_sparse,
                               watcher, log_function):
    task_name = 'InitialConfigurations'
    watcher.start_task(task_name)
    try:
        initial_configurations = ml.create_metalearning_string_for_smac_call(
            configuration_space, basename, metric,
            task, True if is_sparse == 1 else
            False, initial_configurations_via_metalearning,
            metadata_directory)
    except Exception as e:
        log_function(str(e))
        log_function(traceback.format_exc())
        initial_configurations = []
    watcher.stop_task(task_name)
    return initial_configurations


def _print_debug_info_of_init_configuration(initial_configurations, basename,
                                            time_for_task, debug_log, info_log,
                                            watcher):
    debug_log('Initial Configurations: (%d)' % len(initial_configurations))
    for initial_configuration in initial_configurations:
        debug_log(initial_configuration)
    debug_log('Looking for initial configurations took %5.2fsec' %
              watcher.wall_elapsed('InitialConfigurations'))
    info_log(
        'Time left for %s after finding initial configurations: %5.2fsec'
        % (basename, time_for_task -
           watcher.wall_elapsed(basename)))

class AutoML(multiprocessing.Process, BaseEstimator):

    def __init__(self, tmp_dir, output_dir, time_left_for_this_task,
                 per_run_time_limit,
                 log_dir=None,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 seed=1,
                 ml_memory_limit=3000,
                 metadata_directory=None,
                 queue=None,
                 keep_models=True):
        super(AutoML, self).__init__()
        self._tmp_dir = tmp_dir
        self._output_dir = output_dir
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._log_dir = log_dir
        self._initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self._ensemble_size = ensemble_size
        self._ensemble_nbest = ensemble_nbest
        self._seed = seed
        self._ml_memory_limit = ml_memory_limit
        self._metadata_directory = metadata_directory
        self._queue = queue
        self._keep_models = keep_models

        self._basename = None
        self._stopwatch = None
        self._logger = None
        self._ohe = None
        self._task = None
        self._metric = None
        self._target_num = None

        self._debug_mode = False

        self._model_dir = join(self._tmp_dir, 'models_%d' % self._seed)
        self._ensemble_indices_dir = join(self._tmp_dir,
                                          'ensemble_indices_%d' % self._seed)
        self._create_folders()

    def _create_folders(self):
        # == Set up a directory where all the trained models will be pickled to
        os.mkdir(self._model_dir)
        os.mkdir(self._ensemble_indices_dir)

    @staticmethod
    def _save_ensemble_data(x_data, y_data, tmp_dir, watcher):
        """Split dataset and store Data for the ensemble script.

        :param x_data:
        :param y_data:
        :return:

        """
        task_name = 'LoadData'
        watcher.start_task(task_name)
        _, _, _, y_ensemble = split_data.split_data(x_data, y_data)

        filepath = os.path.join(tmp_dir, 'true_labels_ensemble.npy')

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            if not os.path.exists(filepath):
                np.save(filepath, y_ensemble)

        watcher.stop_task(task_name)

    def run(self):
        raise NotImplementedError()

    def fit(self, data_x, y,
            task=MULTICLASS_CLASSIFICATION,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None):
        if dataset_name is None:
            m = hashlib.md5()
            m.update(data_x.data)
            dataset_name = m.hexdigest()

        self._basename = dataset_name

        self._stopwatch = StopWatch()
        self._stopwatch.start_task(self._basename)

        self._logger = _get_logger(self._log_dir, self._basename, self._seed)

        loaded_data_manager = XyDataManager(data_x, y,
                                            task=task,
                                            metric=metric,
                                            feat_type=feat_type,
                                            dataset_name=dataset_name,
                                            encode_labels=False)

        return self._fit(loaded_data_manager)

    def _debug(self, text):
        self._logger.debug(text)

    def _critical(self, text):
        self._logger.critical(text)

    def _error(self, text):
        self._logger.error(text)

    def _info(self, text):
        self._logger.info(text)

    def fit_automl_dataset(self, basename, input_dir):
        # == Creating a data object with data and information about it
        self._basename = basename

        self._stopwatch = StopWatch()
        self._stopwatch.start_task(self._basename)

        self._logger = _get_logger(self._log_dir, self._basename,
                                        self._seed)

        self._debug('======== Reading and converting data ==========')
        # Encoding the labels will be done after the metafeature calculation!
        loaded_data_manager = CompetitionDataManager(self._basename, input_dir,
                                                     verbose=True,
                                                     encode_labels=False)
        loaded_data_manager_str = str(loaded_data_manager).split('\n')
        for part in loaded_data_manager_str:
            self._debug(part)

        return self._fit(loaded_data_manager)

    def _save_data_manager(self, data_d, tmp_dir, basename, watcher):
        task_name = 'StoreDatamanager'

        watcher.start_task(task_name)
        filepath = os.path.join(tmp_dir, basename + '_Manager.pkl')

        if _check_path_for_save(filepath, 'Data manager ', self._debug):
            pickle.dump(data_d, open(filepath, 'w'), protocol=-1)

        watcher.stop_task(task_name)
        return filepath

    @staticmethod
    def _start_task(watcher, task_name):
        watcher.start_task(task_name)

    @staticmethod
    def _stop_task(watcher, task_name):
        watcher.stop_task(task_name)

    @staticmethod
    def _print_load_time(basename, time_left_for_this_task,
                         time_for_load_data, log_function):

        time_left_after_reading = max(
            0, time_left_for_this_task - time_for_load_data)
        log_function('Remaining time after reading %s %5.2f sec' %
                     (basename, time_left_after_reading))
        return time_for_load_data

    def _fit(self, data_d):
        # TODO: check that data and task definition fit together!

        self._metric = data_d.info['metric']
        self._task = data_d.info['task']
        self._target_num = data_d.info['target_num']

        _set_auto_seed(self._seed)

        # load data
        self._save_ensemble_data(
            data_d.data['X_train'],
            data_d.data['Y_train'],
            self._tmp_dir,
            self._stopwatch)

        time_for_load_data = self._stopwatch.wall_elapsed(self._basename)

        if self._debug_mode:
            self._print_load_time(
                self._basename,
                self._time_for_task,
                time_for_load_data,
                self._info)

        # == Calculate metafeatures
        ml = _calculate_metafeatures(
            data_feat_type=data_d.feat_type,
            data_info_task=data_d.info['task'],
            x_train=data_d.data['X_train'],
            y_train=data_d.data['Y_train'],
            basename=self._basename,
            watcher=self._stopwatch,
            metalearning_cnt=self._initial_configurations_via_metalearning,
            log_function=self._debug)

        self._stopwatch.start_task('OneHot')
        data_d.perform1HotEncoding()
        self._ohe = data_d.encoder_
        self._stopwatch.stop_task('OneHot')

        # == Pickle the data manager
        data_manager_path = self._save_data_manager(
            data_d, self._tmp_dir,
                                                    self._basename,
                                                    watcher=self._stopwatch, )

        # = Create a searchspace
        self.configuration_space, configspace_path = _create_search_space(
            self._tmp_dir,
            data_d.info,
            self._stopwatch,
            self._debug)

        if ml is None:
            initial_configurations = []
        elif data_d.info['task'] in [MULTICLASS_CLASSIFICATION,
                                     BINARY_CLASSIFICATION]:

            _calculate_metafeatures_encoded(
                ml,
                self._basename,
                data_d.data['X_train'],
                data_d.data['Y_train'],
                self._stopwatch,
                self._debug)

            self._debug(ml.metafeatures_labels.__repr__(verbosity=2))
            self._debug(ml.metafeatures_encoded_labels.__repr__(verbosity=2))

            initial_configurations = _get_initial_configuration(
                ml,
                self._basename,
                self._metric,
                self.configuration_space,
                self._task,
                self._metadata_directory,
                self._initial_configurations_via_metalearning,
                data_d.info[
                    'is_sparse'],
                self._stopwatch,
                self._error)

            _print_debug_info_of_init_configuration(
                initial_configurations,
                self._basename,
                self._time_for_task,
                self._debug, self._info,
                self._stopwatch)

        else:
            initial_configurations = []
            self._critical('Metafeatures encoded not calculated')

        # == RUN SMAC

        proc_smac = _run_smac(self._tmp_dir, self._basename,
                              self._time_for_task, self._ml_memory_limit,
                              data_manager_path, configspace_path,
                              initial_configurations, self._per_run_time_limit,
                              self._stopwatch, self._debug)

        # == RUN ensemble builder
        proc_ensembles = _run_ensemble_builder(
            self._tmp_dir,
            self._output_dir,
            self._basename,
            self._time_for_task,
            self._task,
            self._metric,
            self._ensemble_size,
            self._ensemble_nbest,
            self._ensemble_indices_dir,
            self._stopwatch,
            self._debug
        )

        if self._queue is not None:
            self._queue.put([time_for_load_data, data_manager_path, proc_smac,
                             proc_ensembles])
        else:
            proc_smac.wait()
            proc_ensembles.wait()

        # Delete AutoSklearn environment variable
        _del_auto_seed()
        return self

    def predict(self, data_x):
        if self._keep_models is not True:
            raise ValueError(
                "Predict can only be called if 'keep_models==True'")

        model_files = os.listdir(self._model_dir)
        models = []
        for model_file in model_files:
            model_file = os.path.join(self._model_dir, model_file)
            with open(model_file) as fh:
                models.append(pickle.load(fh))

        if len(models) == 0:
            raise ValueError('No models fitted!')

        if self._ohe is not None:
            data_x = self._ohe._transform(data_x)

        indices_files = sorted(os.listdir(self._ensemble_indices_dir))
        indices_file = os.path.join(self._ensemble_indices_dir,
                                    indices_files[-1])
        with open(indices_file) as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        predictions = []
        for model, model_file in zip(models, model_files):
            num_run = int(model_file.split('.')[0])

            if num_run not in ensemble_members_run_numbers:
                continue

            weight = ensemble_members_run_numbers[num_run]

            X_ = data_x.copy()
            if self._task in REGRESSION_TASKS:
                prediction = model.predict(X_)
            else:
                prediction = model.predict_proba(X_)
            predictions.append(prediction * weight)

        predictions = np.sum(np.array(predictions), axis=0)
        return predictions

    def score(self, data_x, y):
        prediction = self.predict(data_x)
        return evaluator.calculate_score(y, prediction, self._task,
                                         self._metric, self._target_num)

    def configuration_space_created_hook(self):
        pass

    def get_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'get_params() because it is not intended to '
                                  'be optimized.')

    def set_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'set_params() because it is not intended to '
                                  'be optimized.')
