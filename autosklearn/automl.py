# -*- encoding: utf-8 -*-
from __future__ import print_function

import hashlib
import multiprocessing
import os
import traceback

import numpy as np

from HPOlibConfigSpace.converters import pcs_parser
from sklearn.base import BaseEstimator

from autosklearn.constants import *
from autosklearn.data.data_manager_factory import get_data_manager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.evaluation import resampling, HoldoutEvaluator, get_new_run_num
from autosklearn.metalearning.mismbo import \
    calc_meta_features, calc_meta_features_encoded, \
    create_metalearning_string_for_smac_call
from autosklearn.evaluation import calculate_score
from autosklearn.util import StopWatch, get_logger, setup_logger, \
    get_auto_seed, set_auto_seed, del_auto_seed, submit_process, paramsklearn, \
    Backend
from autosklearn.util.smac import run_smac


def _run_ensemble_builder(tmp_dir,
                          output_dir,
                          basename,
                          time_for_task,
                          task,
                          metric,
                          ensemble_size,
                          ensemble_nbest,
                          watcher,
                          logger):
    if ensemble_size > 0:
        task_name = 'runEnsemble'
        watcher.start_task(task_name)
        time_left_for_ensembles = max(0, time_for_task - watcher.wall_elapsed(
            basename))
        logger.info(
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
        )
        watcher.stop_task(task_name)
        return proc_ensembles
    else:
        logger.info('Not starting ensemble script due to ensemble size 0.')
        return None


def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            metalearning_cnt, x_train, y_train, watcher,
                            logger):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    watcher.start_task(task_name)
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    if metalearning_cnt <= 0:
        result = None
    elif data_info_task in \
            [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION]:
        logger.info('Start calculating metafeatures for %s' % basename)
        result = calc_meta_features(x_train, y_train, categorical=categorical,
                                    dataset_name=basename)
    else:
        result = None
        logger.info('Metafeatures not calculated')
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (categorical attributes) took %5.2f' %
        watcher.wall_elapsed(task_name))
    return result


def _calculate_metafeatures_encoded(basename, x_train, y_train, watcher,
                                    logger):
    task_name = 'CalculateMetafeaturesEncoded'
    watcher.start_task(task_name)
    result = calc_meta_features_encoded(X_train=x_train, Y_train=y_train,
                                        categorical=[False] * x_train.shape[1],
                                        dataset_name=basename)
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec' %
        watcher.wall_elapsed(task_name))
    return result


def _create_search_space(tmp_dir, data_info, backend, watcher, logger,
                         include_estimators=None, include_preprocessors=None):
    task_name = 'CreateConfigSpace'
    watcher.start_task(task_name)
    configspace_path = os.path.join(tmp_dir, 'space.pcs')
    configuration_space = paramsklearn.get_configuration_space(
        data_info,
        include_estimators=include_estimators,
        include_preprocessors=include_preprocessors)
    sp_string = pcs_parser.write(configuration_space)
    backend.write_txt_file(configspace_path, sp_string,
                           'Configuration space')
    watcher.stop_task(task_name)

    return configuration_space, configspace_path


def _get_initial_configuration(meta_features,
                               meta_features_encoded, basename, metric,
                               configuration_space,
                               task, metadata_directory,
                               initial_configurations_via_metalearning,
                               is_sparse,
                               watcher, logger):
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
        logger.error(str(e))
        logger.error(traceback.format_exc())
        initial_configurations = []
    watcher.stop_task(task_name)
    return initial_configurations


def _print_debug_info_of_init_configuration(initial_configurations, basename,
                                            time_for_task, logger, watcher):
    logger.debug('Initial Configurations: (%d)' % len(initial_configurations))
    for initial_configuration in initial_configurations:
        logger.debug(initial_configuration)
    logger.debug('Looking for initial configurations took %5.2fsec' %
                 watcher.wall_elapsed('InitialConfigurations'))
    logger.info(
        'Time left for %s after finding initial configurations: %5.2fsec'
        % (basename, time_for_task - watcher.wall_elapsed(basename)))


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
                 debug_mode=False,
                 include_estimators=None,
                 include_preprocessors=None,
                 resampling_strategy='holdout',
                 resampling_strategy_arguments=None,
                 delete_tmp_dir_after_terminate=false,
                 delete_output_dir_after_terminate=false):
        super(AutoML, self).__init__()

        self._tmp_dir = tmp_dir
        self._output_dir = output_dir
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._log_dir = log_dir if log_dir is not None else self._tmp_dir
        self._initial_configurations_via_metalearning = \
            initial_configurations_via_metalearning
        self._ensemble_size = ensemble_size
        self._ensemble_nbest = ensemble_nbest
        self._seed = seed
        self._ml_memory_limit = ml_memory_limit
        self._metadata_directory = metadata_directory
        self._queue = queue
        self._keep_models = keep_models
        self._include_estimators = include_estimators
        self._include_preprocessors = include_preprocessors
        self._resampling_strategy = resampling_strategy
        self._resampling_strategy_arguments = resampling_strategy_arguments
        self.delete_tmp_dir_after_terminate = delete_tmp_dir_after_terminate
        self.delete_output_dir_after_terminate = \
            delete_output_dir_after_terminate

        self._dataset_name = None
        self._stopwatch = None
        self._logger = None
        self._ohe = None
        self._task = None
        self._metric = None
        self._label_num = None

        self._debug_mode = debug_mode
        self._backend = Backend(self._output_dir, self._tmp_dir)

    def start_automl(self, parser):
        self._backend.save_start_time(self._seed)
        self._stopwatch = StopWatch()
        datamanager = get_data_manager(namespace=parser)
        self._stopwatch.start_task(datamanager.name)

        logger_name = 'AutoML(%d):%s' % (self._seed, datamanager.name)
        setup_logger(os.path.join(self._tmp_dir, '%s.log' % str(logger_name)))
        self._logger = get_logger(logger_name)

        self._datamanager = datamanager
        self._dataset_name = datamanager.name
        self.start()

    def start(self):
        if not hasattr(self, '_datamanager'):
            raise ValueError('You must invoke start() only via start_automl()')
        super(AutoML, self).start()

    def run(self):
        if not hasattr(self, '_datamanager'):
            raise ValueError('You must invoke run() only via start_automl()')
        self._fit(self._datamanager)

    def fit(self, X, y,
            task=MULTICLASS_CLASSIFICATION,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None):
        if dataset_name is None:
            m = hashlib.md5()
            m.update(X.data)
            dataset_name = m.hexdigest()

        self._backend.save_start_time(self._seed)
        self._stopwatch = StopWatch()
        self._dataset_name = dataset_name
        self._stopwatch.start_task(self._dataset_name)

        logger_name = 'AutoML(%d):%s' % (self._seed, dataset_name)
        setup_logger(os.path.join(self._tmp_dir, '%s.log' % str(logger_name)))
        self._logger = get_logger(logger_name)

        loaded_data_manager = XYDataManager(X, y,
                                            task=task,
                                            metric=metric,
                                            feat_type=feat_type,
                                            dataset_name=dataset_name,
                                            encode_labels=False)

        return self._fit(loaded_data_manager)

    def fit_automl_dataset(self, dataset):
        self._stopwatch = StopWatch()
        self._backend.save_start_time(self._seed)

        name = os.path.basename(dataset)
        self._stopwatch.start_task(name)
        self._start_task(self._stopwatch, name)
        self._dataset_name = name

        logger_name = 'AutoML(%d):%s' % (self._seed, name)
        setup_logger(os.path.join(self._tmp_dir, '%s.log' % str(logger_name)))
        self._logger = get_logger(logger_name)

        self._logger.debug('======== Reading and converting data ==========')
        # Encoding the labels will be done after the metafeature calculation!
        loaded_data_manager = CompetitionDataManager(dataset,
                                                     encode_labels=False)
        loaded_data_manager_str = str(loaded_data_manager).split('\n')
        for part in loaded_data_manager_str:
            self._logger.debug(part)

        return self._fit(loaded_data_manager)

    @staticmethod
    def _start_task(watcher, task_name):
        watcher.start_task(task_name)

    @staticmethod
    def _stop_task(watcher, task_name):
        watcher.stop_task(task_name)

    @staticmethod
    def _print_load_time(basename, time_left_for_this_task,
                         time_for_load_data, logger):

        time_left_after_reading = max(
            0, time_left_for_this_task - time_for_load_data)
        logger.info('Remaining time after reading %s %5.2f sec' %
                    (basename, time_left_after_reading))
        return time_for_load_data

    def _do_dummy_prediction(self, datamanager):
        num_run = get_new_run_num(self._tmp_dir)
        he = HoldoutEvaluator(
            datamanager, None,
            with_predictions=True, num_run=num_run,
            output_dir=self._tmp_dir, all_scoring_functions=True)
        he.fit()
        he.file_output()
        model_directory = self._backend.get_model_dir()
        if os.path.exists(model_directory):
            self._backend.save_model(he.model, num_run)
        del he

    def _fit(self, datamanager):
        # Check arguments prior to doing anything!
        if self._resampling_strategy not in ['holdout', 'holdout-iterative-fit',
                                             'cv', 'nested-cv', 'partial-cv']:
            raise ValueError('Illegal resampling strategy: %s' %
                             self._resampling_strategy)
        if self._resampling_strategy == 'partial-cv' and \
                self._ensemble_size != 0:
            raise ValueError("Resampling strategy partial-cv cannot be used "
                             "together with ensembles.")

        self._backend._make_internals_directory()
        if self._keep_models:
            os.mkdir(self._backend.get_model_dir())

        self._metric = datamanager.info['metric']
        self._task = datamanager.info['task']
        self._label_num = datamanager.info['label_num']

        set_auto_seed(self._seed)

        # == Pickle the data manager, here, because no more global
        # OneHotEncoding
        data_manager_path = self._backend.save_datamanager(datamanager)

        self._save_ensemble_data(
            datamanager.data['X_train'],
            datamanager.data['Y_train'])

        time_for_load_data = self._stopwatch.wall_elapsed(self._dataset_name)

        if self._debug_mode:
            self._print_load_time(
                self._dataset_name,
                self._time_for_task,
                time_for_load_data,
                self._logger)

        # == Perform dummy predictions
        self._do_dummy_prediction(datamanager)

        # = Create a searchspace
        # Do this before One Hot Encoding to make sure that it creates a
        # search space for a dense classifier even if one hot encoding would
        # make it sparse (tradeoff; if one hot encoding would make it sparse,
        #  densifier and truncatedSVD would probably lead to a MemoryError,
        # like this we can't use some of the preprocessing methods in case
        # the data became sparse)
        self.configuration_space, configspace_path = _create_search_space(
            self._tmp_dir,
            datamanager.info,
            self._backend,
            self._stopwatch,
            self._logger,
            self._include_estimators,
            self._include_preprocessors)
        self.configuration_space_created_hook(datamanager)

        # == Calculate metafeatures
        meta_features = _calculate_metafeatures(
            data_feat_type=datamanager.feat_type,
            data_info_task=datamanager.info['task'],
            x_train=datamanager.data['X_train'],
            y_train=datamanager.data['Y_train'],
            basename=self._dataset_name,
            watcher=self._stopwatch,
            metalearning_cnt=self._initial_configurations_via_metalearning,
            logger=self._logger)

        self._stopwatch.start_task('OneHot')
        datamanager.perform1HotEncoding()
        self._ohe = datamanager.encoder
        self._stopwatch.stop_task('OneHot')

        if meta_features is None:
            initial_configurations = []
        elif datamanager.info['task'] in [MULTICLASS_CLASSIFICATION,
                                     BINARY_CLASSIFICATION]:

            meta_features_encoded = _calculate_metafeatures_encoded(
                self._dataset_name,
                datamanager.data['X_train'],
                datamanager.data['Y_train'],
                self._stopwatch,
                self._logger)

            self._logger.debug(meta_features.__repr__(verbosity=2))
            self._logger.debug(meta_features_encoded.__repr__(verbosity=2))

            initial_configurations = _get_initial_configuration(
                meta_features,
                meta_features_encoded,
                self._dataset_name,
                self._metric,
                self.configuration_space,
                self._task,
                self._metadata_directory,
                self._initial_configurations_via_metalearning,
                datamanager.info[
                    'is_sparse'],
                self._stopwatch,
                self._logger)

            _print_debug_info_of_init_configuration(
                initial_configurations,
                self._dataset_name,
                self._time_for_task,
                self._logger,
                self._stopwatch)

        else:
            initial_configurations = []
            self._logger('Metafeatures encoded not calculated')

        # == RUN SMAC
        proc_smac = run_smac(self._tmp_dir, self._dataset_name,
                             self._time_for_task, self._ml_memory_limit,
                             data_manager_path, configspace_path,
                             initial_configurations, self._per_run_time_limit,
                             self._stopwatch, self._backend, self._seed,
                             self._resampling_strategy,
                             self._resampling_strategy_arguments)

        # == RUN ensemble builder
        proc_ensembles = _run_ensemble_builder(
            self._tmp_dir,
            self._output_dir,
            self._dataset_name,
            self._time_for_task,
            self._task,
            self._metric,
            self._ensemble_size,
            self._ensemble_nbest,
            self._stopwatch,
            self._logger
        )

        procs = [proc_smac]
        if proc_ensembles is not None:
            procs.append(proc_ensembles)

        if self._queue is not None:
            self._queue.put([time_for_load_data, data_manager_path, procs])
        else:
            for proc in procs:
                proc.wait()

        # Delete AutoSklearn environment variable
        del_auto_seed()

        # In case
        try:
            del self._datamanager
        except Exception:
            pass

        return self

    def predict(self, X):
        if self._keep_models is not True:
            raise ValueError(
                "Predict can only be called if 'keep_models==True'")
        if self._resampling_strategy != 'holdout':
            raise NotImplementedError(
                'Predict is currently only implemented for resampling '
                'strategy holdout.')

        models = self._backend.load_all_models()
        if len(models) == 0:
            raise ValueError('No models fitted!')

        if self._ohe is not None:
            X = self._ohe._transform(X)

        ensemble_indices = self._backend.load_ensemble_indices_weights(
            self._seed)

        predictions = []
        for num_run in models:
            if num_run not in ensemble_indices:
                continue

            weight = ensemble_indices[num_run]
            model = models[num_run]

            X_ = X.copy()
            if self._task in REGRESSION_TASKS:
                prediction = model.predict(X_)
            else:
                prediction = model.predict_proba(X_)
            predictions.append(prediction * weight)

        predictions = np.sum(np.array(predictions), axis=0)
        return predictions

    def score(self, X, y):
        prediction = self.predict(X)
        return calculate_score(y, prediction, self._task,
                               self._metric, self._label_num)

    def _save_ensemble_data(self, X, y):
        """Split dataset and store Data for the ensemble script.

        :param x_data:
        :param y_data:
        :return:

        """
        task_name = 'LoadData'
        self._start_task(self._stopwatch, task_name)
        _, _, _, y_ensemble = resampling.split_data(X, y)
        self._backend.save_targets_ensemble(y_ensemble)
        self._stop_task(self._stopwatch, task_name)

    def configuration_space_created_hook(self, datamanager):
        pass

    def get_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'get_params() because it is not intended to '
                                  'be optimized.')

    def set_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'set_params() because it is not intended to '
                                  'be optimized.')

    def __del__(self):
        self._delete_output_directories()

    def _delete_output_directories(self):
        if self.delete_output_dir_after_terminate:
            try:
                shutil.rmtree(self._output_dir)
            except Exception:
                self._logger.warn("Could not delete output dir: %s" %
                                  self._output_dir)
                pass

        if self.delete_tmp_dir_after_terminate:
            try:
                shutil.rmtree(self._tmp_dir)
            except Exception:
                self._logger.warn("Could not delete tmp dir: %s" %
                              self._tmp_dir)
                pass