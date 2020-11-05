# -*- encoding: utf-8 -*-
import copy
import io
import json
import platform
import os
import sys
import time
from typing import Any, Dict, Optional, List, Union
import unittest.mock
import warnings
import tempfile

from ConfigSpace.read_and_write import json as cs_json
import dask.distributed
import numpy as np
import numpy.ma as ma
import pandas as pd
import pkg_resources
import scipy.stats
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import _RepeatedSplits, \
    BaseShuffleSplit, BaseCrossValidator
from smac.tae import StatusType
from smac.stats.stats import Stats
import joblib
import sklearn.utils
import scipy.sparse
from sklearn.metrics._classification import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.dummy import DummyClassifier, DummyRegressor

from autosklearn.metrics import Scorer
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.data.validation import InputValidator
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.evaluation.abstract_evaluator import _fit_and_suppress_warnings
from autosklearn.evaluation.train_evaluator import _fit_with_budget
from autosklearn.metrics import calculate_score
from autosklearn.util.backend import Backend
from autosklearn.util.stopwatch import StopWatch
from autosklearn.util.logging_ import get_logger, setup_logger
from autosklearn.util import pipeline, RE_PATTERN
from autosklearn.ensemble_builder import EnsembleBuilderManager
from autosklearn.ensembles.singlebest_ensemble import SingleBest
from autosklearn.smbo import AutoMLSMBO
from autosklearn.util.hash import hash_array_or_matrix
from autosklearn.metrics import f1_macro, accuracy, r2
from autosklearn.constants import MULTILABEL_CLASSIFICATION, MULTICLASS_CLASSIFICATION, \
    REGRESSION_TASKS, REGRESSION, BINARY_CLASSIFICATION, MULTIOUTPUT_REGRESSION
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.regression import RegressorChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding import OHEChoice
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense import (
    CoalescenseChoice
)
from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice


def _model_predict(model, X, batch_size, logger, task):
    def send_warnings_to_log(
            message, category, filename, lineno, file=None, line=None):
        logger.debug('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))
        return
    X_ = X.copy()
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        if task in REGRESSION_TASKS:
            if hasattr(model, 'batch_size'):
                prediction = model.predict(X_, batch_size=batch_size)
            else:
                prediction = model.predict(X_)
        else:
            if hasattr(model, 'batch_size'):
                prediction = model.predict_proba(X_, batch_size=batch_size)
            else:
                prediction = model.predict_proba(X_)

            # Check that all probability values lie between 0 and 1.
            assert(
                (prediction >= 0).all() and (prediction <= 1).all()
            ), "For {}, prediction probability not within [0, 1]!".format(
                model
            )

    if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
            X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
        logger.warning(
            "Prediction shape for model %s is %s while X_.shape is %s",
            model, str(prediction.shape), str(X_.shape)
        )
    return prediction


class AutoML(BaseEstimator):

    def __init__(self,
                 backend: Backend,
                 time_left_for_this_task,
                 per_run_time_limit,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 max_models_on_disc=1,
                 seed=1,
                 memory_limit=3072,
                 metadata_directory=None,
                 debug_mode=False,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None,
                 resampling_strategy='holdout-iterative-fit',
                 resampling_strategy_arguments=None,
                 n_jobs=None,
                 dask_client: Optional[dask.distributed.Client] = None,
                 precision=32,
                 disable_evaluator_output=False,
                 get_smac_object_callback=None,
                 smac_scenario_args=None,
                 logging_config=None,
                 metric=None,
                 ):
        super(AutoML, self).__init__()
        self._backend = backend
        # self._tmp_dir = tmp_dir
        # self._output_dir = output_dir
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._initial_configurations_via_metalearning = \
            initial_configurations_via_metalearning
        self._ensemble_size = ensemble_size
        self._ensemble_nbest = ensemble_nbest
        self._max_models_on_disc = max_models_on_disc
        self._seed = seed
        self._memory_limit = memory_limit
        self._data_memory_limit = None
        self._metadata_directory = metadata_directory
        self._include_estimators = include_estimators
        self._exclude_estimators = exclude_estimators
        self._include_preprocessors = include_preprocessors
        self._exclude_preprocessors = exclude_preprocessors
        self._resampling_strategy = resampling_strategy
        self._resampling_strategy_arguments = resampling_strategy_arguments \
            if resampling_strategy_arguments is not None else {}
        if self._resampling_strategy not in ['holdout',
                                             'holdout-iterative-fit',
                                             'cv',
                                             'cv-iterative-fit',
                                             'partial-cv',
                                             'partial-cv-iterative-fit',
                                             ] \
           and not issubclass(self._resampling_strategy, BaseCrossValidator)\
           and not issubclass(self._resampling_strategy, _RepeatedSplits)\
           and not issubclass(self._resampling_strategy, BaseShuffleSplit):
            raise ValueError('Illegal resampling strategy: %s' %
                             self._resampling_strategy)

        if self._resampling_strategy in ['partial-cv',
                                         'partial-cv-iterative-fit',
                                         ] \
           and self._ensemble_size != 0:
            raise ValueError("Resampling strategy %s cannot be used "
                             "together with ensembles." % self._resampling_strategy)
        if self._resampling_strategy in ['partial-cv',
                                         'cv',
                                         'cv-iterative-fit',
                                         'partial-cv-iterative-fit',
                                         ]\
           and 'folds' not in self._resampling_strategy_arguments:
            self._resampling_strategy_arguments['folds'] = 5
        self._n_jobs = n_jobs
        self._dask_client = dask_client

        self.precision = precision
        self._disable_evaluator_output = disable_evaluator_output
        # Check arguments prior to doing anything!
        if not isinstance(self._disable_evaluator_output, (bool, list)):
            raise ValueError('disable_evaluator_output must be of type bool '
                             'or list.')
        if isinstance(self._disable_evaluator_output, list):
            allowed_elements = ['model', 'cv_model', 'y_optimization', 'y_test', 'y_valid']
            for element in self._disable_evaluator_output:
                if element not in allowed_elements:
                    raise ValueError("List member '%s' for argument "
                                     "'disable_evaluator_output' must be one "
                                     "of " + str(allowed_elements))
        self._get_smac_object_callback = get_smac_object_callback
        self._smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config

        self._datamanager = None
        self._dataset_name = None
        self._stopwatch = StopWatch()
        self._logger = None
        self._task = None

        self._metric = metric

        self._label_num = None
        self._parser = None
        self.models_ = None
        self.cv_models_ = None
        self.ensemble_ = None
        self._can_predict = False

        self._debug_mode = debug_mode

        self.InputValidator = InputValidator()

        # The ensemble performance history through time
        self.ensemble_performance_history = []

        if not isinstance(self._time_for_task, int):
            raise ValueError("time_left_for_this_task not of type integer, "
                             "but %s" % str(type(self._time_for_task)))
        if not isinstance(self._per_run_time_limit, int):
            raise ValueError("per_run_time_limit not of type integer, but %s" %
                             str(type(self._per_run_time_limit)))

        # After assigning and checking variables...
        # self._backend = Backend(self._output_dir, self._tmp_dir)

    def _create_dask_client(self):
        self._is_dask_client_internally_created = True
        processes = False
        if self._n_jobs is not None and self._n_jobs > 1:
            processes = True
            dask.config.set({'distributed.worker.daemon': False})
        self._dask_client = dask.distributed.Client(
            dask.distributed.LocalCluster(
                n_workers=self._n_jobs,
                processes=processes,
                threads_per_worker=1,
                # We use the temporal directory to save the
                # dask workers, because deleting workers
                # more time than deleting backend directories
                # This prevent an error saying that the worker
                # file was deleted, so the client could not close
                # the worker properly
                local_directory=tempfile.gettempdir(),
            )
        )

    def _close_dask_client(self):
        if (
            hasattr(self, '_is_dask_client_internally_created')
            and self._is_dask_client_internally_created
            and self._dask_client
        ):
            self._dask_client.shutdown()
            self._dask_client.close()
            del self._dask_client
            self._dask_client = None
            self._is_dask_client_internally_created = False
            del self._is_dask_client_internally_created

    def _get_logger(self, name):
        logger_name = 'AutoML(%d):%s' % (self._seed, name)
        setup_logger(os.path.join(self._backend.temporary_directory,
                                  '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
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

        # When using partial-cv it makes no sense to do dummy predictions
        if self._resampling_strategy in ['partial-cv',
                                         'partial-cv-iterative-fit']:
            return num_run

        self._logger.info("Starting to create dummy predictions.")

        memory_limit = self._memory_limit
        if memory_limit is not None:
            memory_limit = int(memory_limit)

        scenario_mock = unittest.mock.Mock()
        scenario_mock.wallclock_limit = self._time_for_task
        # This stats object is a hack - maybe the SMAC stats object should
        # already be generated here!
        stats = Stats(scenario_mock)
        stats.start_timing()
        ta = ExecuteTaFuncWithQueue(backend=self._backend,
                                    autosklearn_seed=self._seed,
                                    resampling_strategy=self._resampling_strategy,
                                    initial_num_run=num_run,
                                    logger=self._logger,
                                    stats=stats,
                                    metric=self._metric,
                                    memory_limit=memory_limit,
                                    disable_file_output=self._disable_evaluator_output,
                                    abort_on_first_run_crash=False,
                                    cost_for_crash=get_cost_of_crash(self._metric),
                                    **self._resampling_strategy_arguments)

        status, cost, runtime, additional_info = ta.run(num_run, cutoff=self._time_for_task)
        if status == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy predictions.")
        else:
            if additional_info.get('exitcode') == -6:
                self._logger.error(
                    "Dummy prediction failed with run state %s. "
                    "The error suggests that the provided memory limits were too tight. Please "
                    "increase the 'ml_memory_limit' and try again. If this does not solve your "
                    "problem, please open an issue and paste the additional output. "
                    "Additional output: %s.",
                    str(status), str(additional_info),
                )
                # Fail if dummy prediction fails.
                raise ValueError(
                    "Dummy prediction failed with run state %s. "
                    "The error suggests that the provided memory limits were too tight. Please "
                    "increase the 'ml_memory_limit' and try again. If this does not solve your "
                    "problem, please open an issue and paste the additional output. "
                    "Additional output: %s." %
                    (str(status), str(additional_info)),
                )

            else:
                self._logger.error(
                    "Dummy prediction failed with run state %s and additional output: %s.",
                    str(status), str(additional_info),
                )
                # Fail if dummy prediction fails.
                raise ValueError(
                    "Dummy prediction failed with run state %s and additional output: %s."
                    % (str(status), str(additional_info))
                )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: int,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feat_type: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: Optional[bool] = False,
        load_models: bool = True,
    ):
        # Make sure that input is valid
        # Performs Ordinal one hot encoding to the target
        # both for train and test data
        X, y = self.InputValidator.validate(X, y)

        if X_test is not None:
            X_test, y_test = self.InputValidator.validate(X_test, y_test)
            if len(y.shape) != len(y_test.shape):
                raise ValueError('Target value shapes do not match: %s vs %s'
                                 % (y.shape, y_test.shape))

        # Reset learnt stuff
        self.models_ = None
        self.cv_models_ = None
        self.ensemble_ = None

        # The metric must exist as of this point
        # It can be provided in the constructor, or automatically
        # defined in the estimator fit call
        if self._metric is None:
            raise ValueError('No metric given.')
        if not isinstance(self._metric, Scorer):
            raise ValueError('Metric must be instance of '
                             'autosklearn.metrics.Scorer.')

        # If no dask client was provided, we create one, so that we can
        # start a ensemble process in parallel to smbo optimize
        if (
            self._dask_client is None and
            (self._ensemble_size > 0 or self._n_jobs is not None and self._n_jobs > 1)
        ):
            self._create_dask_client()
        else:
            self._is_dask_client_internally_created = False

        if dataset_name is None:
            dataset_name = hash_array_or_matrix(X)

        self._backend.save_start_time(self._seed)
        self._stopwatch = StopWatch()
        self._dataset_name = dataset_name
        self._stopwatch.start_task(self._dataset_name)

        self._logger = self._get_logger(dataset_name)

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

        # Feature types dynamically understood from dataframe
        if feat_type is not None and self.InputValidator.feature_types:
            raise ValueError("feat_type cannot be provided when using pandas "
                             "DataFrame as input. Auto-sklearn extracts the feature types "
                             "automatically from the columns dtypes, so providing feat_type "
                             "not only is not necessary, but not allowed."
                             )
        elif feat_type is None and self.InputValidator.feature_types:
            feat_type = self.InputValidator.feature_types

        # Produce debug information to the logfile
        self._logger.debug('Starting to print environment information')
        self._logger.debug('  Python version: %s', sys.version.split('\n'))
        try:
            self._logger.debug('  Distribution: %s', platform.linux_distribution())
        except AttributeError:
            # platform.linux_distribution() was removed in Python3.8
            # We should move to the distro package as soon as it supports Windows and OSX
            pass
        self._logger.debug('  System: %s', platform.system())
        self._logger.debug('  Machine: %s', platform.machine())
        self._logger.debug('  Platform: %s', platform.platform())
        # UNAME appears to leak sensible information
        # self._logger.debug('  uname: %s', platform.uname())
        self._logger.debug('  Version: %s', platform.version())
        self._logger.debug('  Mac version: %s', platform.mac_ver())
        requirements = pkg_resources.resource_string('autosklearn', 'requirements.txt')
        requirements = requirements.decode('utf-8')
        requirements = [requirement for requirement in requirements.split('\n')]
        for requirement in requirements:
            if not requirement:
                continue
            match = RE_PATTERN.match(requirement)
            if match:
                name = match.group('name')
                module_dist = pkg_resources.get_distribution(name)
                self._logger.debug('  %s', module_dist)
            else:
                raise ValueError('Unable to read requirement: %s' % requirement)
        self._logger.debug('Done printing environment information')
        self._logger.debug('Starting to print arguments to auto-sklearn')
        self._logger.debug('  output_folder: %s', self._backend.context._output_directory)
        self._logger.debug('  tmp_folder: %s', self._backend.context._temporary_directory)
        self._logger.debug('  time_left_for_this_task: %f', self._time_for_task)
        self._logger.debug('  per_run_time_limit: %f', self._per_run_time_limit)
        self._logger.debug(
            '  initial_configurations_via_metalearning: %d',
            self._initial_configurations_via_metalearning,
        )
        self._logger.debug('  ensemble_size: %d', self._ensemble_size)
        self._logger.debug('  ensemble_nbest: %f', self._ensemble_nbest)
        self._logger.debug('  max_models_on_disc: %s', str(self._max_models_on_disc))
        self._logger.debug('  seed: %d', self._seed)
        self._logger.debug('  memory_limit: %s', str(self._memory_limit))
        self._logger.debug('  metadata_directory: %s', self._metadata_directory)
        self._logger.debug('  debug_mode: %s', self._debug_mode)
        self._logger.debug('  include_estimators: %s', str(self._include_estimators))
        self._logger.debug('  exclude_estimators: %s', str(self._exclude_estimators))
        self._logger.debug('  include_preprocessors: %s', str(self._include_preprocessors))
        self._logger.debug('  exclude_preprocessors: %s', str(self._exclude_preprocessors))
        self._logger.debug('  resampling_strategy: %s', str(self._resampling_strategy))
        self._logger.debug('  resampling_strategy_arguments: %s',
                           str(self._resampling_strategy_arguments))
        self._logger.debug('  n_jobs: %s', str(self._n_jobs))
        self._logger.debug('  dask_client: %s', str(self._dask_client))
        self._logger.debug('  precision: %s', str(self.precision))
        self._logger.debug('  disable_evaluator_output: %s', str(self._disable_evaluator_output))
        self._logger.debug('  get_smac_objective_callback: %s', str(self._get_smac_object_callback))
        self._logger.debug('  smac_scenario_args: %s', str(self._smac_scenario_args))
        self._logger.debug('  logging_config: %s', str(self.logging_config))
        self._logger.debug('  metric: %s', str(self._metric))
        self._logger.debug('Done printing arguments to auto-sklearn')
        self._logger.debug('Starting to print available components')
        for choice in (
            ClassifierChoice, RegressorChoice, FeaturePreprocessorChoice,
            OHEChoice, RescalingChoice, CoalescenseChoice,
        ):
            self._logger.debug(
                '%s: %s',
                choice.__name__,
                choice.get_components(),
            )
        self._logger.debug('Done printing available components')

        datamanager = XYDataManager(
            X, y,
            X_test=X_test,
            y_test=y_test,
            task=task,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        self._backend._make_internals_directory()

        self._task = datamanager.info['task']
        self._label_num = datamanager.info['label_num']

        # == Pickle the data manager to speed up loading
        self._backend.save_datamanager(datamanager)

        time_for_load_data = self._stopwatch.wall_elapsed(self._dataset_name)

        if self._debug_mode:
            self._print_load_time(
                self._dataset_name,
                self._time_for_task,
                time_for_load_data,
                self._logger)

        # == Perform dummy predictions
        num_run = 1
        self._do_dummy_prediction(datamanager, num_run)

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
            include_estimators=self._include_estimators,
            exclude_estimators=self._exclude_estimators,
            include_preprocessors=self._include_preprocessors,
            exclude_preprocessors=self._exclude_preprocessors)
        if only_return_configuration_space:
            self._close_dask_client()
            return self.configuration_space

        # == RUN ensemble builder
        # Do this before calculating the meta-features to make sure that the
        # dummy predictions are actually included in the ensemble even if
        # calculating the meta-features takes very long
        ensemble_task_name = 'runEnsemble'
        self._stopwatch.start_task(ensemble_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(self._dataset_name)
        time_left_for_ensembles = max(0, self._time_for_task - elapsed_time)
        proc_ensemble = None
        if time_left_for_ensembles <= 0:
            # Fit only raises error when ensemble_size is not zero but
            # time_left_for_ensembles is zero.
            if self._ensemble_size > 0:
                raise ValueError("Not starting ensemble builder because there "
                                 "is no time left. Try increasing the value "
                                 "of time_left_for_this_task.")
        elif self._ensemble_size <= 0:
            self._logger.info('Not starting ensemble builder because '
                              'ensemble size is <= 0.')
        else:
            self._logger.info(
                'Start Ensemble with %5.2fsec time left' % time_left_for_ensembles)

            proc_ensemble = EnsembleBuilderManager(
                start_time=time.time(),
                time_left_for_ensembles=time_left_for_ensembles,
                backend=copy.deepcopy(self._backend),
                dataset_name=dataset_name,
                task=task,
                metric=self._metric,
                ensemble_size=self._ensemble_size,
                ensemble_nbest=self._ensemble_nbest,
                max_models_on_disc=self._max_models_on_disc,
                seed=self._seed,
                precision=self.precision,
                max_iterations=None,
                read_at_most=np.inf,
                ensemble_memory_limit=self._memory_limit,
                logger_name=self._logger.name,
                random_state=self._seed,
            )

        self._stopwatch.stop_task(ensemble_task_name)

        # kill the datamanager as it will be re-loaded anyways from sub processes
        try:
            del self._datamanager
        except Exception:
            pass

        # => RUN SMAC
        smac_task_name = 'runSMAC'
        self._stopwatch.start_task(smac_task_name)
        elapsed_time = self._stopwatch.wall_elapsed(self._dataset_name)
        time_left_for_smac = max(0, self._time_for_task - elapsed_time)

        if self._logger:
            self._logger.info(
                'Start SMAC with %5.2fsec time left' % time_left_for_smac)
        if time_left_for_smac <= 0:
            self._logger.warning("Not starting SMAC because there is no time "
                                 "left.")
            _proc_smac = None
            self._budget_type = None
        else:
            if self._per_run_time_limit is None or \
                    self._per_run_time_limit > time_left_for_smac:
                self._logger.warning(
                    'Time limit for a single run is higher than total time '
                    'limit. Capping the limit for a single run to the total '
                    'time given to SMAC (%f)' % time_left_for_smac
                )
                per_run_time_limit = time_left_for_smac
            else:
                per_run_time_limit = self._per_run_time_limit

            # Make sure that at least 2 models are created for the ensemble process
            num_models = time_left_for_smac // per_run_time_limit
            if num_models < 2:
                per_run_time_limit = time_left_for_smac//2
                self._logger.warning(
                    "Capping the per_run_time_limit to {} to have "
                    "time for a least 2 models in each process.".format(
                        per_run_time_limit
                    )
                )

            _proc_smac = AutoMLSMBO(
                config_space=self.configuration_space,
                dataset_name=self._dataset_name,
                backend=self._backend,
                total_walltime_limit=time_left_for_smac,
                func_eval_time_limit=per_run_time_limit,
                memory_limit=self._memory_limit,
                data_memory_limit=self._data_memory_limit,
                watcher=self._stopwatch,
                n_jobs=self._n_jobs,
                dask_client=self._dask_client,
                start_num_run=num_run,
                num_metalearning_cfgs=self._initial_configurations_via_metalearning,
                config_file=configspace_path,
                seed=self._seed,
                metadata_directory=self._metadata_directory,
                metric=self._metric,
                resampling_strategy=self._resampling_strategy,
                resampling_strategy_args=self._resampling_strategy_arguments,
                include_estimators=self._include_estimators,
                exclude_estimators=self._exclude_estimators,
                include_preprocessors=self._include_preprocessors,
                exclude_preprocessors=self._exclude_preprocessors,
                disable_file_output=self._disable_evaluator_output,
                get_smac_object_callback=self._get_smac_object_callback,
                smac_scenario_args=self._smac_scenario_args,
                ensemble_callback=proc_ensemble,
            )

            try:
                self.runhistory_, self.trajectory_, self._budget_type = \
                    _proc_smac.run_smbo()
                trajectory_filename = os.path.join(
                    self._backend.get_smac_output_directory_for_run(self._seed),
                    'trajectory.json')
                saveable_trajectory = \
                    [list(entry[:2]) + [entry[2].get_dictionary()] + list(entry[3:])
                     for entry in self.trajectory_]
                with open(trajectory_filename, 'w') as fh:
                    json.dump(saveable_trajectory, fh)
            except Exception as e:
                self._logger.exception(e)
                raise

        # Wait until the ensemble process is finished to avoid shutting down
        # while the ensemble builder tries to access the data
        if proc_ensemble is not None:
            self.ensemble_performance_history = list(proc_ensemble.history)
            if len(proc_ensemble.futures) > 0:
                future = proc_ensemble.futures.pop()
                future.cancel()

        if load_models:
            self._load_models()
        self._close_dask_client()

        return self

    def refit(self, X, y):

        # Make sure input data is valid
        X, y = self.InputValidator.validate(X, y)

        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # Refit is not applicable when ensemble_size is set to zero.
        if self.ensemble_ is None:
            raise ValueError("Refit can only be called if 'ensemble_size != 0'")

        random_state = np.random.RandomState(self._seed)
        for identifier in self.models_:
            model = self.models_[identifier]
            # this updates the model inplace, it can then later be used in
            # predict method

            # try to fit the model. If it fails, shuffle the data. This
            # could alleviate the problem in algorithms that depend on
            # the ordering of the data.
            for i in range(10):
                try:
                    if self._budget_type is None:
                        _fit_and_suppress_warnings(self._logger, model, X, y)
                    else:
                        _fit_with_budget(
                            X_train=X,
                            Y_train=y,
                            budget=identifier[2],
                            budget_type=self._budget_type,
                            logger=self._logger,
                            model=model,
                            train_indices=np.arange(len(X), dtype=int),
                            task_type=self._task,
                        )
                    break
                except ValueError as e:
                    indices = list(range(X.shape[0]))
                    random_state.shuffle(indices)
                    X = X[indices]
                    y = y[indices]

                    if i == 9:
                        raise e

        self._can_predict = True
        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """predict.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the pipelines will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        n_jobs: int, defaults to 1
            Parallelize the predictions across the models with n_jobs
            processes.
        """
        if (
            self._resampling_strategy not in (
                'holdout', 'holdout-iterative-fit', 'cv', 'cv-iterative-fit')
            and not self._can_predict
        ):
            raise NotImplementedError(
                'Predict is currently not implemented for resampling '
                'strategy %s, please call refit().' % self._resampling_strategy)

        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # If self.ensemble_ is None, it means that ensemble_size is set to zero.
        # In such cases, raise error because predict and predict_proba cannot
        # be called.
        if self.ensemble_ is None:
            raise ValueError("Predict and predict_proba can only be called "
                             "if 'ensemble_size != 0'")

        # Make sure that input is valid
        X = self.InputValidator.validate_features(X)

        # Parallelize predictions across models with n_jobs processes.
        # Each process computes predictions in chunks of batch_size rows.
        try:
            for i, tmp_model in enumerate(self.models_.values()):
                if isinstance(tmp_model, (DummyRegressor, DummyClassifier)):
                    check_is_fitted(tmp_model)
                else:
                    check_is_fitted(tmp_model.steps[-1][-1])
            models = self.models_
        except sklearn.exceptions.NotFittedError:
            # When training a cross validation model, self.cv_models_
            # will contain the Voting classifier/regressor product of cv
            # self.models_ in the case of cv, contains unfitted models
            # Raising above exception is a mechanism to detect which
            # attribute contains the relevant models for prediction
            try:
                check_is_fitted(list(self.cv_models_.values())[0])
                models = self.cv_models_
            except sklearn.exceptions.NotFittedError:
                raise ValueError('Found no fitted models!')

        all_predictions = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_model_predict)(
                models[identifier], X, batch_size, self._logger, self._task
            )
            for identifier in self.ensemble_.get_selected_model_identifiers()
        )

        if len(all_predictions) == 0:
            raise ValueError('Something went wrong generating the predictions. '
                             'The ensemble should consist of the following '
                             'models: %s, the following models were loaded: '
                             '%s' % (str(list(self.ensemble_indices_.keys())),
                                     str(list(self.models_.keys()))))

        predictions = self.ensemble_.predict(all_predictions)

        if self._task not in REGRESSION_TASKS:
            # Make sure average prediction probabilities
            # are within a valid range
            # Individual models are checked in _model_predict
            predictions = np.clip(predictions, 0.0, 1.0)

        return predictions

    def fit_ensemble(self, y, task=None, precision=32,
                     dataset_name=None, ensemble_nbest=None,
                     ensemble_size=None):
        if self._resampling_strategy in ['partial-cv', 'partial-cv-iterative-fit']:
            raise ValueError('Cannot call fit_ensemble with resampling '
                             'strategy %s.' % self._resampling_strategy)

        if self._logger is None:
            self._logger = self._get_logger(dataset_name)

        # Make sure that input is valid
        y = self.InputValidator.validate_target(y, is_classification=True)

        # Create a client if needed
        if self._dask_client is None:
            self._create_dask_client()
        else:
            self._is_dask_client_internally_created = False

        # Use the current thread to start the ensemble builder process
        # The function ensemble_builder_process will internally create a ensemble
        # builder in the provide dask client
        manager = EnsembleBuilderManager(
            start_time=time.time(),
            time_left_for_ensembles=self._time_for_task,
            backend=copy.deepcopy(self._backend),
            dataset_name=dataset_name if dataset_name else self._dataset_name,
            task=task if task else self._task,
            metric=self._metric,
            ensemble_size=ensemble_size if ensemble_size else self._ensemble_size,
            ensemble_nbest=ensemble_nbest if ensemble_nbest else self._ensemble_nbest,
            max_models_on_disc=self._max_models_on_disc,
            seed=self._seed,
            precision=precision if precision else self.precision,
            max_iterations=1,
            read_at_most=np.inf,
            ensemble_memory_limit=self._memory_limit,
            random_state=self._seed,
            logger_name=self._logger.name,
        )
        manager.build_ensemble(self._dask_client)
        future = manager.futures.pop()
        dask.distributed.wait([future])  # wait for the ensemble process to finish
        result = future.result()
        if result is None:
            raise ValueError("Error building the ensemble - please check the log file and command "
                             "line output for error messages.")
        self.ensemble_performance_history, _, _, _, _ = result

        self._load_models()
        self._close_dask_client()
        return self

    def _load_models(self):
        self.ensemble_ = self._backend.load_ensemble(self._seed)

        # If no ensemble is loaded, try to get the best performing model
        if not self.ensemble_:
            self.ensemble_ = self._load_best_individual_model()

        if self.ensemble_:
            identifiers = self.ensemble_.get_selected_model_identifiers()
            self.models_ = self._backend.load_models_by_identifiers(identifiers)
            if self._resampling_strategy in ('cv', 'cv-iterative-fit'):
                self.cv_models_ = self._backend.load_cv_models_by_identifiers(identifiers)
            else:
                self.cv_models_ = None
            if (
                len(self.models_) == 0 and
                self._resampling_strategy not in ['partial-cv', 'partial-cv-iterative-fit']
            ):
                raise ValueError('No models fitted!')
            if (
                self._resampling_strategy in ['cv', 'cv-iterative-fit']
                and len(self.cv_models_) == 0
            ):
                raise ValueError('No models fitted!')

        elif self._disable_evaluator_output is False or \
                (isinstance(self._disable_evaluator_output, list) and
                 'model' not in self._disable_evaluator_output):
            model_names = self._backend.list_all_models(self._seed)

            if len(model_names) == 0 and self._resampling_strategy not in \
                    ['partial-cv', 'partial-cv-iterative-fit']:
                raise ValueError('No models fitted!')

            self.models_ = []

        else:
            self.models_ = []

    def _load_best_individual_model(self):
        """
        In case of failure during ensemble building,
        this method returns the single best model found
        by AutoML.
        This is a robust mechanism to be able to predict,
        even though no ensemble was found by ensemble builder.
        """

        # We also require that the model is fit and a task is defined
        # The ensemble size must also be greater than 1, else it means
        # that the user intentionally does not want an ensemble
        if not self._task or self._ensemble_size < 1:
            return None

        # SingleBest contains the best model found by AutoML
        ensemble = SingleBest(
            metric=self._metric,
            seed=self._seed,
            run_history=self.runhistory_,
            backend=self._backend,
        )
        self._logger.warning(
            "No valid ensemble was created. Please check the log"
            "file for errors. Default to the best individual estimator:{}".format(
                ensemble.identifiers_
            )
        )
        return ensemble

    def score(self, X, y):
        # fix: Consider only index 1 of second dimension
        # Don't know if the reshaping should be done there or in calculate_score

        # Predict has validate within it, so we
        # call it before the upcoming validate call
        # The reason is we do not want to trigger the
        # check for changing input types on successive
        # input validator calls
        prediction = self.predict(X)

        # Make sure that input is valid
        X, y = self.InputValidator.validate(X, y)

        # Encode the prediction using the input validator
        # We train autosklearn with a encoded version of y,
        # which is decoded by predict().
        # Above call to validate() encodes the y given for score()
        # Below call encodes the prediction, so we compare in the
        # same representation domain
        prediction = self.InputValidator.encode_target(prediction)

        return calculate_score(solution=y,
                               prediction=prediction,
                               task_type=self._task,
                               metric=self._metric,
                               all_scoring_functions=False)

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

        # TODO remove this restriction!
        if self._resampling_strategy in ['partial-cv', 'partial-cv-iterative-fit']:
            raise ValueError('Cannot call cv_results when using partial-cv!')

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
        budgets = []
        for run_key in self.runhistory_.data:
            run_value = self.runhistory_.data[run_key]
            config_id = run_key.config_id
            config = self.runhistory_.ids_config[config_id]

            s = run_value.status
            if s == StatusType.SUCCESS:
                status.append('Success')
            elif s == StatusType.DONOTADVANCE:
                status.append('Success (but do not advance to higher budget)')
            elif s == StatusType.TIMEOUT:
                status.append('Timeout')
            elif s == StatusType.CRASHED:
                status.append('Crash')
            elif s == StatusType.ABORT:
                status.append('Abort')
            elif s == StatusType.MEMOUT:
                status.append('Memout')
            elif s == StatusType.RUNNING:
                continue
            elif s == StatusType.BUDGETEXHAUSTED:
                continue
            else:
                raise NotImplementedError(s)

            param_dict = config.get_dictionary()
            params.append(param_dict)
            mean_test_score.append(self._metric._optimum - (self._metric._sign * run_value.cost))
            mean_fit_time.append(run_value.time)
            budgets.append(run_key.budget)

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
        results['budgets'] = budgets

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
        sio.write('  Metric: %s\n' % self._metric)
        idx_success = np.where(np.array(
            [status in ['Success', 'Success (but do not advance to higher budget)']
             for status in cv_results['status']]
        ))[0]
        if len(idx_success) > 0:
            if not self._metric._optimum:
                idx_best_run = np.argmin(cv_results['mean_test_score'][idx_success])
            else:
                idx_best_run = np.argmax(cv_results['mean_test_score'][idx_success])
            best_score = cv_results['mean_test_score'][idx_success][idx_best_run]
            sio.write('  Best validation score: %f\n' % best_score)
        num_runs = len(cv_results['status'])
        sio.write('  Number of target algorithm runs: %d\n' % num_runs)
        num_success = sum([
            s in ['Success', 'Success (but do not advance to higher budget)']
            for s in cv_results['status']
        ])
        sio.write('  Number of successful target algorithm runs: %d\n' % num_success)
        num_crash = sum([s == 'Crash' for s in cv_results['status']])
        sio.write('  Number of crashed target algorithm runs: %d\n' % num_crash)
        num_timeout = sum([s == 'Timeout' for s in cv_results['status']])
        sio.write('  Number of target algorithms that exceeded the time '
                  'limit: %d\n' % num_timeout)
        num_memout = sum([s == 'Memout' for s in cv_results['status']])
        sio.write('  Number of target algorithms that exceeded the memory '
                  'limit: %d\n' % num_memout)
        return sio.getvalue()

    def get_models_with_weights(self):
        if self.models_ is None or len(self.models_) == 0 or \
                self.ensemble_ is None:
            self._load_models()

        return self.ensemble_.get_models_with_weights(self.models_)

    def show_models(self):
        models_with_weights = self.get_models_with_weights()

        with io.StringIO() as sio:
            sio.write("[")
            for weight, model in models_with_weights:
                sio.write("(%f, %s),\n" % (weight, model))
            sio.write("]")

            return sio.getvalue()

    def _create_search_space(self, tmp_dir, backend, datamanager,
                             include_estimators=None,
                             exclude_estimators=None,
                             include_preprocessors=None,
                             exclude_preprocessors=None):
        task_name = 'CreateConfigSpace'

        self._stopwatch.start_task(task_name)
        configspace_path = os.path.join(tmp_dir, 'space.json')
        configuration_space = pipeline.get_configuration_space(
            datamanager.info,
            include_estimators=include_estimators,
            exclude_estimators=exclude_estimators,
            include_preprocessors=include_preprocessors,
            exclude_preprocessors=exclude_preprocessors)
        configuration_space = self.configuration_space_created_hook(
            datamanager, configuration_space)
        backend.write_txt_file(
            configspace_path,
            cs_json.write(configuration_space),
            'Configuration space'
        )
        self._stopwatch.stop_task(task_name)

        return configuration_space, configspace_path

    def configuration_space_created_hook(self, datamanager, configuration_space):
        return configuration_space

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a client!
        self._dask_client = None
        return self.__dict__

    def __del__(self):
        self._close_dask_client()

        # When a multiprocessing work is done, the
        # objects are deleted. We don't want to delete run areas
        # until the estimator is deleted
        self._backend.context.delete_directories(force=False)


class AutoMLClassifier(AutoML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._task_mapping = {'multilabel-indicator': MULTILABEL_CLASSIFICATION,
                              'multiclass': MULTICLASS_CLASSIFICATION,
                              'binary': BINARY_CLASSIFICATION}

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ):

        # We first validate the dtype of the target provided by the user
        # In doing so, we also fit the internal encoder for classification
        # In case y_test is provided, we proactively check their type
        # and make sure the enconding accounts for both y_test/y_train classes
        input_y = self.InputValidator.join_and_check(y, y_test) if y_test is not None else y
        y_task = type_of_target(
            self.InputValidator.validate_target(input_y, is_classification=True)
        )
        task = self._task_mapping.get(y_task)
        if task is None:
            raise ValueError('Cannot work on data of type %s' % y_task)

        if self._metric is None:
            if task == MULTILABEL_CLASSIFICATION:
                self._metric = f1_macro
            else:
                self._metric = accuracy

        return super().fit(
            X, y,
            X_test=X_test,
            y_test=y_test,
            task=task,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
        )

    def predict(self, X, batch_size=None, n_jobs=1):
        predicted_probabilities = super().predict(X, batch_size=batch_size,
                                                  n_jobs=n_jobs)

        if self.InputValidator.is_single_column_target() == 1:
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
        else:
            predicted_indexes = (predicted_probabilities > 0.5).astype(int)

        return self.InputValidator.decode_target(predicted_indexes)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)


class AutoMLRegressor(AutoML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_mapping = {'continuous-multioutput': MULTIOUTPUT_REGRESSION,
                              'continuous': REGRESSION,
                              'multiclass': REGRESSION}

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ):

        # Check the data provided in y
        # After the y data type is validated,
        # check the task type
        y_task = type_of_target(
            self.InputValidator.validate_target(y)
        )
        task = self._task_mapping.get(y_task)
        if task is None:
            raise ValueError('Cannot work on data of type %s' % y_task)
        if self._metric is None:
            self._metric = r2

        return super().fit(
            X, y,
            X_test=X_test,
            y_test=y_test,
            task=task,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
        )
