# -*- encoding: utf-8 -*-
from __future__ import print_function

import hashlib
import multiprocessing
import os
import sys
import shutil
import traceback
import pynisher

import numpy as np

from ConfigSpace.io import pcs
from ConfigSpace.configuration_space import Configuration
from sklearn.base import BaseEstimator
import six

from autosklearn.constants import *
from autosklearn.data.data_manager_factory import get_data_manager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.evaluation import resampling, HoldoutEvaluator, get_new_run_num
from autosklearn.evaluation import calculate_score
from autosklearn.util import StopWatch, get_logger, setup_logger, \
    get_auto_seed, set_auto_seed, del_auto_seed, pipeline, \
    Backend
from autosklearn.ensemble_builder import main as ensemble_main
#from autosklearn.util.smac import run_smac
from autosklearn.smbo import AutoMLSMBO

def _create_search_space(tmp_dir, data_info, backend, watcher, logger,
                         include_estimators=None, include_preprocessors=None):
    task_name = 'CreateConfigSpace'
    watcher.start_task(task_name)
    configspace_path = os.path.join(tmp_dir, 'space.pcs')
    configuration_space = pipeline.get_configuration_space(
        data_info,
        include_estimators=include_estimators,
        include_preprocessors=include_preprocessors)
    sp_string = pcs.write(configuration_space)
    backend.write_txt_file(configspace_path, sp_string,
                           'Configuration space')
    watcher.stop_task(task_name)

    return configuration_space, configspace_path

class EnsembleProcess(multiprocessing.Process):

    """
    Wrap the ensemble script into its own process.
    """
    
    def __init__(self, tmp_dir, dataset_name, task_type, metric, limit,
                 output_dir, ensemble_size, ensemble_nbest, seed,
                 shared_mode, precision, max_iterations = None, silent=True):
        super(EnsembleProcess, self).__init__()

        self.tmp_dir = tmp_dir
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.metric = metric
        self.limit = limit
        self.output_dir = output_dir
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.seed = seed
        self.shared_mode = shared_mode
        self.max_iterations = max_iterations
        self.precision = precision
        self.silent = silent
        if self.max_iterations is None:
            self.max_iterations = -1
        if self.limit <= 0:
            self.limit = None
        else:
            self.limit = max(1, self.limit)

    def run(self):
        if self.silent:
            f = open(os.devnull, 'w')
            sys.stdout = f
            sys.stderr = f
        # TODO LIMITS
        safe_ensemble_script = pynisher.enforce_limits()(ensemble_main)
        safe_ensemble_script(autosklearn_tmp_dir = self.tmp_dir,
                             dataset_name = self.dataset_name,
                             task_type = self.task_type,
                             metric = self.metric,
                             limit = self.limit,
                             output_dir = self.output_dir,
                             ensemble_size = self.ensemble_size,
                             ensemble_nbest = self.ensemble_nbest,
                             seed = self.seed,
                             shared_mode = self.shared_mode,
                             max_iterations = self.max_iterations,
                             precision = self.precision)

class AutoML(BaseEstimator, multiprocessing.Process):

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
                 resampling_strategy='holdout-iterative-fit',
                 resampling_strategy_arguments=None,
                 delete_tmp_folder_after_terminate=False,
                 delete_output_folder_after_terminate=False,
                 shared_mode=False,
                 precision=32,
                 max_iter_smac=None):
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
        self._max_iter_smac = max_iter_smac
        self.delete_tmp_folder_after_terminate = \
            delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = \
            delete_output_folder_after_terminate
        self._shared_mode = shared_mode
        self.precision = precision

        self._datamanager = None
        self._dataset_name = None
        self._stopwatch = StopWatch()
        self._logger = None
        self._task = None
        self._metric = None
        self._label_num = None
        self._paraser = None
        self.models_ = None
        self.ensemble_indices_ = None

        self._debug_mode = debug_mode
        self._backend = Backend(self._output_dir, self._tmp_dir)

    def start_automl(self, parser):
        self._parser = parser
        self.start()

    def start(self):
        if self._parser is None:
            raise ValueError('You must invoke start() only via start_automl()')
        super(AutoML, self).start()

    def run(self):
        if self._parser is None:
            raise ValueError('You must invoke run() only via start_automl()')
        self._backend.save_start_time(self._seed)
        self._stopwatch = StopWatch()
        datamanager = get_data_manager(namespace=self._parser)
        self._stopwatch.start_task(datamanager.name)

        logger_name = 'AutoML(%d):%s' % (self._seed, datamanager.name)
        setup_logger(os.path.join(self._tmp_dir, '%s.log' % str(logger_name)))
        self._logger = get_logger(logger_name)

        self._datamanager = datamanager
        self._dataset_name = datamanager.name
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

        if isinstance(metric, str):
            metric = STRING_TO_METRIC[metric]

        if feat_type is not None and len(feat_type) != X.shape[1]:
            raise ValueError('Array feat_type does not have same number of '
                             'variables as X has features. %d vs %d.' %
                             (len(feat_type), X.shape[1]))
        if feat_type is not None and not all([isinstance(f, bool)
                                              for f in feat_type]):
            raise ValueError('Array feat_type must only contain bools.')

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
        loaded_data_manager = CompetitionDataManager(
            dataset, encode_labels=False,
            max_memory_in_mb=float(self._ml_memory_limit) / 3)
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

    def _do_dummy_prediction(self, datamanager, num_run):
        def dummy_prediction_call(datamanager):
            out_dir = self._tmp_dir
            evaluator = HoldoutEvaluator(datamanager,
                                         output_dir=out_dir,
                                         configuration=None,
                                         seed=1,
                                         num_run=0,
                                         with_predictions=True,
                                         all_scoring_functions=True,
                                         output_y_test=True)
            evaluator.fit()
            evaluator.finish_up()
            backend = Backend(None, out_dir)
            if os.path.exists(backend.get_model_dir()):
                backend.save_model(evaluator.model, num_run, 1)
    
        self._logger.info("Starting to create dummy predictions.")
        # TODO enforce limits
        safe_call = pynisher.enforce_limits()(dummy_prediction_call)
        try:
            safe_call(datamanager)
        except:
            pass
        self._logger.info("Finished creating dummy predictions.")

    def _fit(self, datamanager):
        # Reset learnt stuff
        self.models_ = None
        self.ensemble_indices_ = None

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
            try:
                os.mkdir(self._backend.get_model_dir())
            except OSError:
                self._logger.warning("model directory already exists")
                if not self._shared_mode:
                    raise

        self._metric = datamanager.info['metric']
        self._task = datamanager.info['task']
        self._label_num = datamanager.info['label_num']

        set_auto_seed(self._seed)

        # == Pickle the data manager to speed up loading
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
        num_run = 1
        if self._resampling_strategy in ['holdout', 'holdout-iterative-fit']:
            self._do_dummy_prediction(datamanager, num_run)

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

        # == RUN ensemble builder
        # Do this before calculating the meta-features to make sure that the
        # dummy predictions are actually included in the ensemble even if
        # calculating the meta-features takes very long
        ensemble_task_name = 'runEnsemble'
        self._stopwatch.start_task(ensemble_task_name)
        time_left_for_ensembles = max(0,self._time_for_task \
                                      - self._stopwatch.wall_elapsed(self._dataset_name))
        if self._logger:
            self._logger.info(
                'Start Ensemble with %5.2fsec time left' % time_left_for_ensembles)
        if time_left_for_ensembles <= 0:
            logger.warning("Not starting ensemble builder because it's not worth it")
            self._proc_ensemble = None
        else:
            self._proc_ensemble = EnsembleProcess(
                tmp_dir=self._tmp_dir,
                dataset_name=self._dataset_name,
                task_type=self._task,
                metric=self._metric,
                limit=time_left_for_ensembles,
                output_dir=self._output_dir,
                ensemble_size=self._ensemble_size,
                ensemble_nbest=self._ensemble_nbest,
                seed=self._seed,
                shared_mode=self._shared_mode,
                precision=self.precision,
                # JTS TODO: enable silent again and check that it works
                silent=False 
            )
            self._proc_ensemble.start()
        self._stopwatch.stop_task(ensemble_task_name)

        # == RUN SMBO
        default_configs = []
        # == set default configurations
        if (datamanager.info["task"] == BINARY_CLASSIFICATION) or \
            (datamanager.info["task"] == MULTICLASS_CLASSIFICATION):
            config = {'balancing:strategy': 'weighting',
                      'classifier:__choice__': 'sgd',
                      'classifier:sgd:loss': 'hinge',
                      'classifier:sgd:penalty': 'l2',
                      'classifier:sgd:alpha': 0.0001,
                      'classifier:sgd:fit_intercept': 'True',
                      'classifier:sgd:n_iter': 5,
                      'classifier:sgd:learning_rate': 'optimal',
                      'classifier:sgd:eta0': 0.01,
                      'classifier:sgd:average': 'True',
                      'imputation:strategy': 'mean',
                      'one_hot_encoding:use_minimum_fraction': 'True',
                      'one_hot_encoding:minimum_fraction': 0.1,
                      'preprocessor:__choice__': 'no_preprocessing',
                      'rescaling:__choice__': 'min/max'}
            default_configs.append(Configuration(self.configuration_space, config))
        elif datamanager.info["task"] == MULTILABEL_CLASSIFICATION:
            config = {'classifier:__choice__': 'adaboost',
                      'classifier:adaboost:algorithm': 'SAMME.R',
                      'classifier:adaboost:learning_rate': 1.0,
                      'classifier:adaboost:max_depth': 1,
                      'classifier:adaboost:n_estimators': 50,
                      'balancing:strategy': 'weighting',
                      'imputation:strategy': 'mean',
                      'one_hot_encoding:use_minimum_fraction': 'True',
                      'one_hot_encoding:minimum_fraction': 0.1,
                      'preprocessor:__choice__': 'no_preprocessing',
                      'rescaling:__choice__': 'none'}
            default_configs.append(Configuration(self.configuration_space, config))
        else:
            config = None
            self._logger.info("Tasktype unknown: %s" %
                              TASK_TYPES_TO_STRING[datamanager.info["task"]])

        # kill the datamanager as it will be re-loaded anyways from sub processes
        try:
            del self._datamanager
        except Exception:
            pass
            
        # => RUN SMAC
        # JTS TODO: check difference between cutoff time and limit
        self._proc_smac = AutoMLSMBO(config_space = self.configuration_space,
                                     dataset_name = self._dataset_name,
                                     tmp_dir = self._tmp_dir,
                                     output_dir = self._output_dir,
                                     limit = self._time_for_task,
                                     cutoff_time = self._time_for_task,
                                     memory_limit = self._ml_memory_limit,
                                     logger = self._logger,
                                     watcher = self._stopwatch,
                                     start_num_run = num_run,
                                     default_cfgs = default_configs,
                                     num_metalearning_cfgs = self._initial_configurations_via_metalearning,
                                     config_file = configspace_path,
                                     smac_iters = self._max_iter_smac,
                                     metadata_directory=self._metadata_directory)
        self._proc_smac.start()

        procs = []
        if self._proc_smac is not None:
            procs.append(self._proc_smac)
        if self._proc_ensemble is not None:
            procs.append(self._proc_ensemble)

        if self._queue is not None:
            self._queue.put([time_for_load_data, data_manager_path, procs])
        else:
            for proc in procs:
                proc.join()

        # Delete AutoSklearn environment variable
        del_auto_seed()

        if self._queue is None:
            self._load_models()

        return self

    def predict(self, X):
        if self._keep_models is not True:
            raise ValueError(
                "Predict can only be called if 'keep_models==True'")
        if self._resampling_strategy not in  ['holdout',
                                              'holdout-iterative-fit']:
            raise NotImplementedError(
                'Predict is currently only implemented for resampling '
                'strategy holdout.')

        if self.models_ is None or len(self.models_) == 0 or len(
                self.ensemble_indices_) == 0:
            self._load_models()

        predictions = []
        for identifier in self.models_:
            if identifier not in self.ensemble_indices_:
                continue

            weight = self.ensemble_indices_[identifier]
            model = self.models_[identifier]

            X_ = X.copy()
            if self._task in REGRESSION_TASKS:
                prediction = model.predict(X_)
            else:
                prediction = model.predict_proba(X_)

            if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
                    X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
                self._logger.warning("Prediction shape for model %s is %s "
                                     "while X_.shape is %s" %
                                     (model, str(prediction.shape),
                                      str(X_.shape)))
            predictions.append(prediction * weight)

        if len(predictions) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_indices_.keys())),
                                     str(list(self.models_.keys()))))

        predictions = np.sum(np.array(predictions), axis=0)
        return predictions

    def _load_models(self):
        if self._shared_mode:
            seed = -1
        else:
            seed = self._seed

        self.models_ = self._backend.load_all_models(seed)
        if len(self.models_) == 0:
            raise ValueError('No models fitted!')

        self.ensemble_indices_ = self._backend.load_ensemble_indices_weights(
            seed)

    def score(self, X, y):
        # fix: Consider only index 1 of second dimension
        # Don't know if the reshaping should be done there or in calculate_score
        prediction = self.predict(X)
        if self._task == BINARY_CLASSIFICATION:
            prediction = prediction[:, 1].reshape((-1, 1))
        return calculate_score(y, prediction, self._task,
                               self._metric, self._label_num,
                               logger=self._logger)

    def show_models(self):
        if self.models_ is None or len(self.models_) == 0 or len(
                self.ensemble_indices_) == 0:
            self._load_models()

        output = []
        sio = six.StringIO()
        for identifier in self.models_:
            if identifier not in self.ensemble_indices_:
                continue

            weight = self.ensemble_indices_[identifier]
            model = self.models_[identifier]
            output.append((weight, model))

        output.sort(reverse=True)

        sio.write("[")
        for weight, model in output:
            sio.write("(%f, %s),\n" % (weight, model))
        sio.write("]")

        return sio.getvalue()

    def _save_ensemble_data(self, X, y):
        """Split dataset and store Data for the ensemble script.

        :param X:
        :param y:
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
        if self.delete_output_folder_after_terminate:
            try:
                shutil.rmtree(self._output_dir)
            except Exception:
                if self._logger is not None:
                    self._logger.warning("Could not delete output dir: %s" %
                                         self._output_dir)
                else:
                    print("Could not delete output dir: %s" %
                          self._output_dir)

        if self.delete_tmp_folder_after_terminate:
            try:
                shutil.rmtree(self._tmp_dir)
            except Exception:
                if self._logger is not None:
                    self._logger.warning("Could not delete tmp dir: %s" %
                                  self._tmp_dir)
                    pass
                else:
                    print("Could not delete tmp dir: %s" %
                          self._tmp_dir)
