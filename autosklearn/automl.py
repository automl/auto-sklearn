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
from autosklearn.constants import MULTICLASS_CLASSIFICATION, REGRESSION_TASKS
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data_managers import CompetitionDataManager, XYDataManager
from autosklearn.evaluators import calculate_score
from autosklearn.metalearning import \
    calc_meta_features, calc_meta_features_encoded, \
    create_metalearning_string_for_smac_call
from autosklearn.util import StopWatch, get_logger, get_auto_seed, \
    set_auto_seed, del_auto_seed, \
    add_file_handler, submit_process, split_data, paramsklearn


def _save_ensemble_data(x_data, y_data, tmp_dir, watcher):
    """Split dataset and store Data for the ensemble script.

    :param x_data:
    :param y_data:
    :return:

    """
    task_name = 'LoadData'
    watcher.start_task(task_name)
    _, _, _, y_ensemble = split_data(x_data, y_data)

    filepath = os.path.join(tmp_dir, 'true_labels_ensemble.npy')

    lock_path = filepath + '.lock'
    with lockfile.LockFile(lock_path):
        if not os.path.exists(filepath):
            np.save(filepath, y_ensemble)

    watcher.stop_task(task_name)

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
    logger = get_logger(os.path.basename(__file__))
    logger_file = os.path.join(log_dir, '%s.log' % str(
        'AutoML_%s_%d' % (basename, seed)))
    add_file_handler(logger, logger_file)
    return logger



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
                                seed=get_auto_seed())
    log_function(smac_call)
    watcher.stop_task(task_name)
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
    time_left_for_ensembles = max(0, time_for_task - watcher.wall_elapsed(
        basename))
    log_function(
        'Start Ensemble with %5.2fsec time left' % time_left_for_ensembles)
    proc_ensembles = submit_process.run_ensemble_builder(
        tmp_dir=tmp_dir,
        dataset_name=basename,
        task_type=task,
        metric=metric,
        limit=time_left_for_ensembles,
        output_dir=output_dir,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        seed=get_auto_seed(),
        ensemble_indices_output_dir=ensemble_indices_dir
    )
    watcher.stop_task(task_name)
    return proc_ensembles


def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            metalearning_cnt, x_train, y_train, watcher,
                            log_function):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    watcher.start_task(task_name)
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    if metalearning_cnt <= 0:
        result = None
    elif data_info_task in \
            [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION]:
        # todo
        # теперь нет класса, поэтому заменить везде на функции
        log_function('Start calculating metafeatures for %s' % basename)
        result = calc_meta_features(x_train, y_train, categorical=categorical,
                                    dataset_name=basename)
    else:
        result = None
        log_function('Metafeatures not calculated')
    watcher.stop_task(task_name)
    log_function(
        'Calculating Metafeatures (categorical attributes) took %5.2f' %
        watcher.wall_elapsed(task_name))
    return result


def _calculate_metafeatures_encoded(basename, x_train, y_train, watcher,
                                    log_funciton):
    task_name = 'CalculateMetafeaturesEncoded'
    watcher.start_task(task_name)
    result = calc_meta_features_encoded(X_train=x_train, Y_train=y_train,
                                        categorical=[False] * x_train.shape[0],
                                        dataset_name=basename)
    watcher.stop_task(task_name)
    log_funciton(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec' %
        watcher.wall_elapsed(task_name))
    return result

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


def _get_initial_configuration(meta_features,
                               meta_features_encoded, basename, metric,
                               configuration_space,
                               task, metadata_directory,
                               initial_configurations_via_metalearning,
                               is_sparse,
                               watcher, log_function):
    task_name = 'InitialConfigurations'
    watcher.start_task(task_name)
    try:
        initial_configurations = create_metalearning_string_for_smac_call(
            meta_features,
            meta_features_encoded,
            configuration_space, basename, metric,
            task,
            is_sparse == 1,
            initial_configurations_via_metalearning,
            metadata_directory
        )
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

    def __init__(self,
                 tmp_dir,
                 output_dir,
                 time_left_for_this_task,
                 per_run_time_limit,
                 log_dir=None,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 seed=1,
                 ml_memory_limit=3000,
                 metadata_directory=None,
                 queue=None,
                 keep_models=True,
                 debug_mode=False):
        super(AutoML, self).__init__()
        self._tmp_dir = tmp_dir
        self._output_dir = output_dir
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._log_dir = log_dir if log_dir is not None else self._tmp_dir
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

        self._debug_mode = debug_mode

        self._model_dir = join(self._tmp_dir, 'models_%d' % self._seed)
        self._ensemble_indices_dir = join(self._tmp_dir,
                                          'ensemble_indices_%d' % self._seed)
        self._create_folders()

    def _create_folders(self):
        # == Set up a directory where all the trained models will be pickled to
        os.mkdir(self._model_dir)
        os.mkdir(self._ensemble_indices_dir)

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

        loaded_data_manager = XYDataManager(data_x, y,
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

    def _fit(self, manager):
        # TODO: check that data and task definition fit together!

        self._metric = manager.info['metric']
        self._task = manager.info['task']
        self._target_num = manager.info['target_num']

        set_auto_seed(self._seed)

        # load data
        _save_ensemble_data(
            manager.data['X_train'],
            manager.data['Y_train'],
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
        meta_features = _calculate_metafeatures(
            data_feat_type=manager.feat_type,
            data_info_task=manager.info['task'],
            x_train=manager.data['X_train'],
            y_train=manager.data['Y_train'],
            basename=self._basename,
            watcher=self._stopwatch,
            metalearning_cnt=self._initial_configurations_via_metalearning,
            log_function=self._debug)

        self._stopwatch.start_task('OneHot')
        manager.perform_hot_encoding()
        self._ohe = manager.encoder_
        self._stopwatch.stop_task('OneHot')

        # == Pickle the data manager
        data_manager_path = self._save_data_manager(
            manager,
            self._tmp_dir,
            self._basename,
            watcher=self._stopwatch)

        # = Create a searchspace
        self.configuration_space, configspace_path = _create_search_space(
            self._tmp_dir,
            manager.info,
            self._stopwatch,
            self._debug)

        if meta_features is None:
            initial_configurations = []
        elif manager.info['task'] in [MULTICLASS_CLASSIFICATION,
                                     BINARY_CLASSIFICATION]:

            meta_features_encoded = _calculate_metafeatures_encoded(
                self._basename,
                manager.data['X_train'],
                manager.data['Y_train'],
                self._stopwatch,
                self._debug)

            self._debug(meta_features.__repr__(verbosity=2))
            self._debug(meta_features_encoded.__repr__(verbosity=2))

            initial_configurations = _get_initial_configuration(
                meta_features,
                meta_features_encoded,
                self._basename,
                self._metric,
                self.configuration_space,
                self._task,
                self._metadata_directory,
                self._initial_configurations_via_metalearning,
                manager.info[
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
        del_auto_seed()
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
        return calculate_score(y, prediction, self._task,
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
