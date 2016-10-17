# -*- encoding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
import hashlib
import io
import os


from ConfigSpace.io import pcs
import numpy as np
import numpy.ma as ma
import scipy.stats
from sklearn.base import BaseEstimator
from smac.tae.execute_ta_run import StatusType
from sklearn.grid_search import _CVScoreTuple

from autosklearn.constants import *
from autosklearn.data.data_manager_factory import get_data_manager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.evaluation import resampling, eval_with_limits
from autosklearn.evaluation import calculate_score
from autosklearn.util import StopWatch, get_logger, setup_logger, \
    pipeline
from autosklearn.ensemble_builder import EnsembleBuilder
from autosklearn.smbo import AutoMLSMBO


class AutoML(BaseEstimator):

    def __init__(self,
                 backend,
                 time_left_for_this_task,
                 per_run_time_limit,
                 log_dir=None,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 seed=1,
                 ml_memory_limit=3000,
                 metadata_directory=None,
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
                 max_iter_smac=None,
                 acquisition_function='EI'):
        super(AutoML, self).__init__()
        self._backend = backend
        #self._tmp_dir = tmp_dir
        #self._output_dir = output_dir
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        #self._log_dir = log_dir if log_dir is not None else self._tmp_dir
        self._initial_configurations_via_metalearning = \
            initial_configurations_via_metalearning
        self._ensemble_size = ensemble_size
        self._ensemble_nbest = ensemble_nbest
        self._seed = seed
        self._ml_memory_limit = ml_memory_limit
        self._data_memory_limit = None
        self._metadata_directory = metadata_directory
        self._keep_models = keep_models
        self._include_estimators = include_estimators
        self._include_preprocessors = include_preprocessors
        self._resampling_strategy = resampling_strategy
        self._resampling_strategy_arguments = resampling_strategy_arguments
        self._max_iter_smac = max_iter_smac
        #self.delete_tmp_folder_after_terminate = \
        #    delete_tmp_folder_after_terminate
        #self.delete_output_folder_after_terminate = \
        #    delete_output_folder_after_terminate
        self._shared_mode = shared_mode
        self.precision = precision
        self.acquisition_function = acquisition_function

        self._datamanager = None
        self._dataset_name = None
        self._stopwatch = StopWatch()
        self._logger = None
        self._task = None
        self._metric = None
        self._label_num = None
        self._parser = None
        self.models_ = None
        self.ensemble_ = None
        self._can_predict = False

        self._debug_mode = debug_mode

        if not isinstance(self._time_for_task, int):
            raise ValueError("time_left_for_this_task not of type integer, "
                             "but %s" % str(type(self._time_for_task)))
        if not isinstance(self._per_run_time_limit, int):
            raise ValueError("per_run_time_limit not of type integer, but %s" %
                             str(type(self._per_run_time_limit)))

        # After assignging and checking variables...
        #self._backend = Backend(self._output_dir, self._tmp_dir)

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

        self._logger = self._get_logger(datamanager.name)

        self._datamanager = datamanager
        self._dataset_name = datamanager.name
        self._fit(self._datamanager)

    def fit(self, X, y,
            task=MULTICLASS_CLASSIFICATION,
            metric='acc_metric',
            feat_type=None,
            dataset_name=None):
        if not self._shared_mode:
            self._backend.context.delete_directories()
        else:
            # If this fails, it's likely that this is the first call to get
            # the data manager
            try:
                D = self._backend.load_datamanager()
                dataset_name = D.name
            except IOError:
                pass

        self._backend.context.create_directories()

        if dataset_name is None:
            m = hashlib.md5()
            m.update(X.data)
            dataset_name = m.hexdigest()

        self._backend.save_start_time(self._seed)
        self._stopwatch = StopWatch()
        self._dataset_name = dataset_name
        self._stopwatch.start_task(self._dataset_name)

        self._logger = self._get_logger(dataset_name)

        if isinstance(metric, str):
            metric = STRING_TO_METRIC[metric]

        if feat_type is not None and len(feat_type) != X.shape[1]:
            raise ValueError('Array feat_type does not have same number of '
                             'variables as X has features. %d vs %d.' %
                             (len(feat_type), X.shape[1]))
        if feat_type is not None and not all([isinstance(f, str)
                                              for f in feat_type]):
            raise ValueError('Array feat_type must only contain strings.')
        if feat_type is not None:
            for ft in feat_type:
                if ft.lower() not in ['categorical', 'numerical']:
                    raise ValueError('Only `Categorical` and `Numerical` are '
                                     'valid feature types, you passed `%s`' % ft)

        self._data_memory_limit = None
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

        self._logger = self._get_logger(name)
        self._logger.debug('======== Reading and converting data ==========')
        # Encoding the labels will be done after the metafeature calculation!
        self._data_memory_limit = float(self._ml_memory_limit) / 3
        loaded_data_manager = CompetitionDataManager(
            dataset, encode_labels=False,
            max_memory_in_mb=self._data_memory_limit)
        loaded_data_manager_str = str(loaded_data_manager).split('\n')
        for part in loaded_data_manager_str:
            self._logger.debug(part)

        return self._fit(loaded_data_manager)

    def _get_logger(self, name):
        logger_name = 'AutoML(%d):%s' % (self._seed, name)
        setup_logger(os.path.join(self._backend.temporary_directory, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

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

        self._logger.info("Starting to create dummy predictions.")
        time_limit = int(self._time_for_task / 6.)
        memory_limit = int(self._ml_memory_limit)

        _info = eval_with_limits(datamanager, self._backend, 1,
                                 self._seed, num_run,
                                 self._resampling_strategy,
                                 self._resampling_strategy_arguments,
                                 memory_limit, time_limit,
                                 logger=self._logger)
        if _info[4] == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy prediction 1/2.")
        else:
            self._logger.error('Error creating dummy prediction 1/2:%s ',
                               _info[3])

        num_run += 1

        _info = eval_with_limits(datamanager, self._backend, 2,
                                 self._seed, num_run,
                                 self._resampling_strategy,
                                 self._resampling_strategy_arguments,
                                 memory_limit, time_limit,
                                 logger=self._logger)
        if _info[4] == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy prediction 2/2.")
        else:
            self._logger.error('Error creating dummy prediction 2/2 %s',
                               _info[3])

        num_run += 1
        return num_run

    def _fit(self, datamanager):
        # Reset learnt stuff
        self.models_ = None
        self.ensemble_ = None

        # Check arguments prior to doing anything!
        if self._resampling_strategy not in ['holdout', 'holdout-iterative-fit',
                                             'cv', 'nested-cv', 'partial-cv']:
            raise ValueError('Illegal resampling strategy: %s' %
                             self._resampling_strategy)
        if self._resampling_strategy == 'partial-cv' and \
                self._ensemble_size != 0:
            raise ValueError("Resampling strategy partial-cv cannot be used "
                             "together with ensembles.")

        acquisition_functions = ['EI', 'EIPS']
        if self.acquisition_function not in acquisition_functions:
            raise ValueError('Illegal acquisition %s: Must be one of %s.' %
                             (self.acquisition_function, acquisition_functions))

        self._backend._make_internals_directory()
        if self._keep_models:
            try:
                os.mkdir(self._backend.get_model_dir())
            except OSError:
                if not self._shared_mode:
                    raise

        self._metric = datamanager.info['metric']
        self._task = datamanager.info['task']
        self._label_num = datamanager.info['label_num']

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
        #if self._resampling_strategy in ['holdout', 'holdout-iterative-fit']:
        num_run = self._do_dummy_prediction(datamanager, num_run)

        # = Create a searchspace
        # Do this before One Hot Encoding to make sure that it creates a
        # search space for a dense classifier even if one hot encoding would
        # make it sparse (tradeoff; if one hot encoding would make it sparse,
        #  densifier and truncatedSVD would probably lead to a MemoryError,
        # like this we can't use some of the preprocessing methods in case
        # the data became sparse)
        self.configuration_space, configspace_path = self._create_search_space(
            self._backend.temporary_directory,
            self._backend,
            datamanager,
            self._include_estimators,
            self._include_preprocessors)

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
            self._logger.warning("Not starting ensemble builder because there "
                                 "is no time left!")
            self._proc_ensemble = None
        else:
            self._proc_ensemble = self._get_ensemble_process(time_left_for_ensembles)
            if self._ensemble_size > 0:
                self._proc_ensemble.start()
            else:
                self._logger.info('Not starting ensemble builder because '
                                  'ensemble size is <= 0.')
        self._stopwatch.stop_task(ensemble_task_name)

        # kill the datamanager as it will be re-loaded anyways from sub processes
        try:
            del self._datamanager
        except Exception:
            pass
            
        # => RUN SMAC
        smac_task_name = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        time_left_for_smac = max(0,
            self._time_for_task - self._stopwatch.wall_elapsed(
            self._dataset_name))

        if self._logger:
            self._logger.info(
                'Start SMAC with %5.2fsec time left' % time_left_for_smac)
        if time_left_for_smac <= 0:
            self._logger.warning("Not starting SMAC because there is no time "
                                 "left.")
            self._proc_smac = None
        else:
            if self._per_run_time_limit is None or \
                    self._per_run_time_limit > time_left_for_smac:
                print('Time limit for a single run is higher than total time '
                      'limit. Capping the limit for a single run to the total '
                      'time given to SMAC (%f)' % time_left_for_smac)
                per_run_time_limit = time_left_for_smac
            else:
                per_run_time_limit = self._per_run_time_limit

            self._proc_smac = AutoMLSMBO(config_space=self.configuration_space,
                                         dataset_name=self._dataset_name,
                                         backend=self._backend,
                                         total_walltime_limit=time_left_for_smac,
                                         func_eval_time_limit=per_run_time_limit,
                                         memory_limit=self._ml_memory_limit,
                                         data_memory_limit=self._data_memory_limit,
                                         watcher=self._stopwatch,
                                         start_num_run=num_run,
                                         num_metalearning_cfgs=self._initial_configurations_via_metalearning,
                                         config_file=configspace_path,
                                         smac_iters=self._max_iter_smac,
                                         seed=self._seed,
                                         metadata_directory=self._metadata_directory,
                                         resampling_strategy=self._resampling_strategy,
                                         resampling_strategy_args=self._resampling_strategy_arguments,
                                         acquisition_function=self.acquisition_function,
                                         shared_mode=self._shared_mode)
            self._proc_smac.run_smbo()

        self._proc_ensemble = None
        self._load_models()

        return self

    def refit(self, X, y):
        if self._keep_models is not True:
            raise ValueError(
                "Predict can only be called if 'keep_models==True'")
        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        for identifier in self.models_:
            if identifier in self.ensemble_.get_model_identifiers():
                model = self.models_[identifier]
                # this updates the model inplace, it can then later be used in
                # predict method
                model.fit(X.copy(), y.copy())

        self._can_predict = True
        return self

    def predict(self, X):
        if self._keep_models is not True:
            raise ValueError(
                "Predict can only be called if 'keep_models==True'")
        if not self._can_predict and \
                self._resampling_strategy not in  \
                        ['holdout', 'holdout-iterative-fit']:
            raise NotImplementedError(
                'Predict is currently only implemented for resampling '
                'strategy holdout.')

        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        all_predictions = []
        for identifier in self.ensemble_.get_model_identifiers():
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
            all_predictions.append(prediction)

        if len(all_predictions) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_indices_.keys())),
                                     str(list(self.models_.keys()))))

        predictions = self.ensemble_.predict(all_predictions)
        return predictions

    def fit_ensemble(self, y, task=None, metric=None, precision='32',
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        if self._logger is None:
            self._logger = self._get_logger(dataset_name)

        self._proc_ensemble = self._get_ensemble_process(
            1, task, metric, precision, dataset_name, max_iterations=1,
            ensemble_nbest=ensemble_nbest, ensemble_size=ensemble_size)
        self._proc_ensemble.main()
        return self

    def _get_ensemble_process(self, time_left_for_ensembles,
                              task=None, metric=None, precision=None,
                              dataset_name=None, max_iterations=-1,
                              ensemble_nbest=None, ensemble_size=None):

        if task is None:
            task = self._task
        else:
            self._task = task

        if metric is None:
            metric = self._metric
        else:
            self._metric = metric

        if precision is None:
            precision = self.precision
        else:
            self.precision = precision

        if dataset_name is None:
            dataset_name = self._dataset_name
        else:
            self._dataset_name = dataset_name

        if ensemble_nbest is None:
            ensemble_nbest = self._ensemble_nbest
        else:
            self._ensemble_nbest = ensemble_nbest

        if ensemble_size is None:
            ensemble_size = self._ensemble_size
        else:
            self._ensemble_size = ensemble_size

        return EnsembleBuilder(backend=self._backend,
                               dataset_name=dataset_name,
                               task_type=task,
                               metric=metric,
                               limit=time_left_for_ensembles,
                               ensemble_size=ensemble_size,
                               ensemble_nbest=ensemble_nbest,
                               seed=self._seed,
                               shared_mode=self._shared_mode,
                               precision=precision,
                               max_iterations=max_iterations)

    def _load_models(self):
        if self._shared_mode:
            seed = -1
        else:
            seed = self._seed

        self.ensemble_ = self._backend.load_ensemble(seed)
        if self.ensemble_:
            identifiers = self.ensemble_.identifiers_
            self.models_ = self._backend.load_models_by_identifiers(identifiers)
        else:
            self.models_ = self._backend.load_all_models(seed)

        if len(self.models_) == 0:
            raise ValueError('No models fitted!')

    def score(self, X, y):
        # fix: Consider only index 1 of second dimension
        # Don't know if the reshaping should be done there or in calculate_score
        prediction = self.predict(X)
        return calculate_score(y, prediction, self._task,
                               self._metric, self._label_num,
                               logger=self._logger)

    @property
    def grid_scores_(self):
        grid_scores = list()

        scores_per_config = defaultdict(list)
        config_list = list()

        for run_key in self._proc_smac.runhistory.data:
            run_value = self._proc_smac.runhistory.data[run_key]

            config_id = run_key.config_id
            cost = run_value.cost

            if config_id not in config_list:
                config_list.append(config_id)

            scores_per_config[config_id].append(cost)

        for config_id in config_list:
            scores = [1 - score for score in scores_per_config[config_id]]
            mean_score = np.mean(scores)
            config = self._proc_smac.runhistory.ids_config[config_id]

            grid_score = _CVScoreTuple(config.get_dictionary(), mean_score,
                                       scores)
            grid_scores.append(grid_score)

        return grid_scores

    @property
    def cv_results_(self):
        results = dict()

        # Missing in contrast to scikit-learn
        # splitX_test_score - auto-sklearn does not store the scores on a split
        #                     basis
        # std_test_score - auto-sklearn does not store the scores on a split
        #                  basis
        # splitX_train_score - auto-sklearn does not compute train scores, add
        #                      flag to compute the train scores
        # mean_train_score - auto-sklearn does not store the train scores
        # std_train_score - auto-sklearn does not store the train scores
        # std_fit_time - auto-sklearn does not store the fit times per split
        # mean_score_time - auto-sklearn does not store the score time
        # std_score_time - auto-sklearn does not store the score time
        # TODO: add those arguments

        parameter_dictionaries = dict()
        masks = dict()
        hp_names = []

        # Set up dictionary for parameter values
        for hp in self.configuration_space.get_hyperparameters():
            name = hp.name
            parameter_dictionaries[name] = []
            masks[name] = []
            hp_names.append(name)

        mean_test_score = []
        mean_fit_time = []
        params = []
        status = []
        for run_key in self._proc_smac.runhistory.data:
            run_value = self._proc_smac.runhistory.data[run_key]
            config_id = run_key.config_id
            config = self._proc_smac.runhistory.ids_config[config_id]

            param_dict = config.get_dictionary()
            params.append(param_dict)
            mean_test_score.append(1 - run_value.cost)
            mean_fit_time.append(run_value.time)
            s = run_value.status
            if s == 1:
                status.append('Success')
            elif s == 2:
                status.append('Timeout')
            elif s == 3:
                status.append('Crash')
            elif s == 4:
                status.append('Abort')
            elif s == 5:
                status.append('Memout')
            else:
                status.append('Unknown')

            for hp_name in hp_names:
                if hp_name in param_dict:
                    hp_value = param_dict[hp_name]
                    mask_value = False
                else:
                    hp_value = np.NaN
                    mask_value = True

                parameter_dictionaries[hp_name].append(hp_value)
                masks[hp_name].append(mask_value)

        results['mean_test_score'] = np.array(mean_test_score)
        results['mean_fit_time'] = np.array(mean_fit_time)
        results['params'] = params
        results['rank_test_scores'] = scipy.stats.rankdata(1 - results['mean_test_score'],
                                                           method='min')
        results['status'] = status

        for hp_name in hp_names:
            masked_array = ma.MaskedArray(parameter_dictionaries[hp_name],
                                          masks[hp_name])
            results['param_%s' % hp_name] = masked_array

        return results

    def sprint_statistics(self):
        cv_results = self.cv_results_
        sio = io.StringIO()
        sio.write('auto-sklearn results:\n')
        sio.write('  Dataset name: %s\n' % self._dataset_name)
        sio.write('  Metric: %s\n' % METRIC_TO_STRING[self._metric])
        idx_best_run = np.argmax(cv_results['mean_test_score'])
        best_score = cv_results['mean_test_score'][idx_best_run]
        sio.write('  Best validation score: %f\n' % best_score)
        num_runs = len(cv_results['status'])
        sio.write('  Number of target algorithm runs: %d\n' % num_runs)
        num_success = sum([s == 'Success' for s in cv_results['status']])
        sio.write('  Number of successful target algorithm runs: %d\n' % num_success)
        num_crash = sum([s == 'Crash' for s in cv_results['status']])
        sio.write('  Number of crashed target algorithm runs: %d\n' % num_crash)
        num_timeout = sum([s == 'Timeout' for s in cv_results['status']])
        sio.write('  Number of target algorithms that exceeded the memory '
                  'limit: %d\n' % num_timeout)
        num_memout = sum([s == 'Memout' for s in cv_results['status']])
        sio.write('  Number of target algorithms that exceeded the time '
                  'limit: %d\n' % num_memout)
        return sio.getvalue()


    def show_models(self):
        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        return self.ensemble_.pprint_ensemble_string(self.models_)

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

    def _create_search_space(self, tmp_dir, backend, datamanager,
                             include_estimators=None,
                             include_preprocessors=None):
        task_name = 'CreateConfigSpace'

        self._stopwatch.start_task(task_name)
        configspace_path = os.path.join(tmp_dir, 'space.pcs')
        configuration_space = pipeline.get_configuration_space(
            datamanager.info,
            include_estimators=include_estimators,
            include_preprocessors=include_preprocessors)
        configuration_space = self.configuration_space_created_hook(
            datamanager, configuration_space)
        sp_string = pcs.write(configuration_space)
        backend.write_txt_file(configspace_path, sp_string,
                               'Configuration space')
        self._stopwatch.stop_task(task_name)

        return configuration_space, configspace_path

    def configuration_space_created_hook(self, datamanager, configuration_space):
        return configuration_space
