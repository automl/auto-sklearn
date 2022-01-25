# -*- encoding: utf-8 -*-
import copy
import distro
import io
import json
import platform
import logging.handlers
import multiprocessing
import os
import sys
import time
from typing import Any, Dict, Optional, List, Tuple, Union
import uuid
import unittest.mock
import tempfile

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as cs_json
import dask
import dask.distributed
import numpy as np
import numpy.ma as ma
import pandas as pd
import pkg_resources
import scipy.stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection._split import _RepeatedSplits, \
    BaseShuffleSplit, BaseCrossValidator
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType
from smac.stats.stats import Stats
import joblib
import sklearn.utils
from scipy.sparse import spmatrix
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics._classification import type_of_target
from sklearn.dummy import DummyClassifier, DummyRegressor

from autosklearn.automl_common.common.utils.backend import Backend, create

from autosklearn.metrics import Scorer, default_metric_for_task
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.data.validation import (
    convert_if_sparse,
    InputValidator,
    SUPPORTED_FEAT_TYPES,
    SUPPORTED_TARGET_TYPES,
)
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.evaluation.abstract_evaluator import _fit_and_suppress_warnings
from autosklearn.evaluation.train_evaluator import TrainEvaluator, _fit_with_budget
from autosklearn.metrics import calculate_metric
from autosklearn.util.stopwatch import StopWatch
from autosklearn.util.logging_ import (
    setup_logger,
    start_log_server,
    get_named_client_logger,
    warnings_to,
    PicklableClientLogger,
)
from autosklearn.util import pipeline, RE_PATTERN
from autosklearn.util.parallel import preload_modules
from autosklearn.ensemble_builder import EnsembleBuilderManager
from autosklearn.ensembles.singlebest_ensemble import SingleBest
from autosklearn.smbo import AutoMLSMBO
from autosklearn.constants import MULTILABEL_CLASSIFICATION, MULTICLASS_CLASSIFICATION, \
    REGRESSION_TASKS, REGRESSION, BINARY_CLASSIFICATION, MULTIOUTPUT_REGRESSION, \
    CLASSIFICATION_TASKS
from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.regression import RegressorChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding import OHEChoice
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense import (
    CoalescenseChoice
)
from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
from autosklearn.util.single_thread_client import SingleThreadedClient


def _model_predict(
    model: Any,
    X: SUPPORTED_FEAT_TYPES,
    task: int,
    batch_size: Optional[int] = None,
    logger: Optional[PicklableClientLogger] = None,
) -> np.ndarray:
    """ Generates the predictions from a model.

    This is seperated out into a seperate function to allow for multiprocessing
    and perform parallel predictions.

    Parameters
    ----------
    model: Any
        The model to perform predictions with

    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to perform predictions on.

    task: int
        The int identifier indicating the kind of task that the model was
        trained on.

    batchsize: Optional[int] = None
        If the model supports batch_size predictions then it's possible to pass
        this in as an argument.

    logger: Optional[PicklableClientLogger] = None
        If a logger is passed, the warnings are writte to the logger. Otherwise
        the warnings propogate as they would normally.

    Returns
    -------
    np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        The predictions produced by the model
    """
    # Copy the array and ensure is has the attr 'shape'
    X_ = np.asarray(X) if isinstance(X, list) else X.copy()

    assert X_.shape[0] >= 1, f"X must have more than 1 sample but has {X_.shape[0]}"

    with warnings_to(logger=logger):
        # TODO issue 1169
        #   VotingRegressors aren't meant to be used for multioutput but we are
        #   using them anyways. Hence we need to manually get their outputs and
        #   average the right index as it averages on wrong dimension for us.
        #   We should probaly move away from this in the future.
        #
        #   def VotingRegressor.predict()
        #       return np.average(self._predict(X), axis=1) <- wrong axis
        #
        if task == MULTIOUTPUT_REGRESSION and isinstance(model, VotingRegressor):
            voting_regressor = model
            prediction = np.average(voting_regressor.transform(X_), axis=2).T

        else:
            if task in CLASSIFICATION_TASKS:
                predict_func = model.predict_proba
            else:
                predict_func = model.predict

            if batch_size is not None and hasattr(model, 'batch_size'):
                prediction = predict_func(X_, batch_size=batch_size)
            else:
                prediction = predict_func(X_)

    # Check that probability values lie between 0 and 1.
    if task in CLASSIFICATION_TASKS:
        assert (prediction >= 0).all() and (prediction <= 1).all(), \
            f"For {model}, prediction probability not within [0, 1]!"

    assert prediction.shape[0] == X_.shape[0], \
        f"Prediction shape {model} is {prediction.shape} while X_.shape is {X_.shape}"

    return prediction


class AutoML(BaseEstimator):

    def __init__(self,
                 time_left_for_this_task,
                 per_run_time_limit,
                 temporary_directory: Optional[str] = None,
                 delete_tmp_folder_after_terminate: bool = True,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 max_models_on_disc=1,
                 seed=1,
                 memory_limit=3072,
                 metadata_directory=None,
                 debug_mode=False,
                 include: Optional[Dict[str, List[str]]] = None,
                 exclude: Optional[Dict[str, List[str]]] = None,
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
                 scoring_functions=None,
                 get_trials_callback=None
                 ):
        super(AutoML, self).__init__()
        self.configuration_space = None
        self._backend: Optional[Backend] = None
        self._temporary_directory = temporary_directory
        self._delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        # self._tmp_dir = tmp_dir
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
        self._include = include
        self._exclude = exclude
        self._resampling_strategy = resampling_strategy
        self._scoring_functions = scoring_functions if scoring_functions is not None else []
        self._resampling_strategy_arguments = resampling_strategy_arguments \
            if resampling_strategy_arguments is not None else {}
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
        self._get_trials_callback = get_trials_callback
        self._smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config

        self._datamanager = None
        self._dataset_name = None
        self._feat_type = None
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

        self.InputValidator = None  # type: Optional[InputValidator]

        # The ensemble performance history through time
        self.ensemble_performance_history = []

        # Single core, local runs should use fork
        # to prevent the __main__ requirements in
        # examples. Nevertheless, multi-process runs
        # have spawn as requirement to reduce the
        # possibility of a deadlock
        self._multiprocessing_context = 'forkserver'
        if self._n_jobs == 1 and self._dask_client is None:
            self._multiprocessing_context = 'fork'
            self._dask_client = SingleThreadedClient()

        if not isinstance(self._time_for_task, int):
            raise ValueError("time_left_for_this_task not of type integer, "
                             "but %s" % str(type(self._time_for_task)))
        if not isinstance(self._per_run_time_limit, int):
            raise ValueError("per_run_time_limit not of type integer, but %s" %
                             str(type(self._per_run_time_limit)))

        # By default try to use the TCP logging port or get a new port
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        # Num_run tell us how many runs have been launched
        # It can be seen as an identifier for each configuration
        # saved to disk
        self.num_run = 0

    def _create_backend(self) -> Backend:
        return create(
            temporary_directory=self._temporary_directory,
            output_directory=None,
            prefix="auto-sklearn",
            delete_tmp_folder_after_terminate=self._delete_tmp_folder_after_terminate,
        )

    def _create_dask_client(self):
        self._is_dask_client_internally_created = True
        self._dask_client = dask.distributed.Client(
            dask.distributed.LocalCluster(
                n_workers=self._n_jobs,
                processes=False,
                threads_per_worker=1,
                # We use the temporal directory to save the
                # dask workers, because deleting workers
                # more time than deleting backend directories
                # This prevent an error saying that the worker
                # file was deleted, so the client could not close
                # the worker properly
                local_directory=tempfile.gettempdir(),
                # Memory is handled by the pynisher, not by the dask worker/nanny
                memory_limit=0,
            ),
            # Heartbeat every 10s
            heartbeat_interval=10000,
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

        # Setup the configuration for the logger
        # This is gonna be honored by the server
        # Which is created below
        setup_logger(
            filename='%s.log' % str(logger_name),
            logging_config=self.logging_config,
            output_dir=self._backend.temporary_directory,
        )

        # As Auto-sklearn works with distributed process,
        # we implement a logger server that can receive tcp
        # pickled messages. They are unpickled and processed locally
        # under the above logging configuration setting
        # We need to specify the logger_name so that received records
        # are treated under the logger_name ROOT logger setting
        context = multiprocessing.get_context(self._multiprocessing_context)
        preload_modules(context)
        self.stop_logging_server = context.Event()
        port = context.Value('l')  # be safe by using a long
        port.value = -1

        self.logging_server = context.Process(
            target=start_log_server,
            kwargs=dict(
                host='localhost',
                logname=logger_name,
                event=self.stop_logging_server,
                port=port,
                filename='%s.log' % str(logger_name),
                logging_config=self.logging_config,
                output_dir=self._backend.temporary_directory,
            ),
        )

        self.logging_server.start()

        while True:
            with port.get_lock():
                if port.value == -1:
                    time.sleep(0.01)
                else:
                    break

        self._logger_port = int(port.value)

        return get_named_client_logger(
            name=logger_name,
            host='localhost',
            port=self._logger_port,
        )

    def _clean_logger(self):
        if not hasattr(self, 'stop_logging_server') or self.stop_logging_server is None:
            return

        # Clean up the logger
        if self.logging_server.is_alive():
            self.stop_logging_server.set()

            # We try to join the process, after we sent
            # the terminate event. Then we try a join to
            # nicely join the event. In case something
            # bad happens with nicely trying to kill the
            # process, we execute a terminate to kill the
            # process.
            self.logging_server.join(timeout=5)
            self.logging_server.terminate()
            del self.stop_logging_server

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

    def _do_dummy_prediction(self, datamanager: XYDataManager, num_run: int) -> int:

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
                                    stats=stats,
                                    metric=self._metric,
                                    memory_limit=memory_limit,
                                    disable_file_output=self._disable_evaluator_output,
                                    abort_on_first_run_crash=False,
                                    cost_for_crash=get_cost_of_crash(self._metric),
                                    port=self._logger_port,
                                    pynisher_context=self._multiprocessing_context,
                                    **self._resampling_strategy_arguments)

        status, cost, runtime, additional_info = ta.run(num_run, cutoff=self._time_for_task)
        if status == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy predictions.")
        else:
            if additional_info.get('exitcode') == -6:
                self._logger.error(
                    "Dummy prediction failed with run state %s. "
                    "The error suggests that the provided memory limits were too tight. Please "
                    "increase the 'memory_limit' and try again. If this does not solve your "
                    "problem, please open an issue and paste the additional output. "
                    "Additional output: %s.",
                    str(status), str(additional_info),
                )
                # Fail if dummy prediction fails.
                raise ValueError(
                    "Dummy prediction failed with run state %s. "
                    "The error suggests that the provided memory limits were too tight. Please "
                    "increase the 'memory_limit' and try again. If this does not solve your "
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
        return num_run

    @classmethod
    def _task_type_id(cls, task_type: str) -> int:
        raise NotImplementedError

    @classmethod
    def _supports_task_type(cls, task_type: str) -> bool:
        raise NotImplementedError

    def fit(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Union[SUPPORTED_TARGET_TYPES, spmatrix],
        task: Optional[int] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[Union[SUPPORTED_TARGET_TYPES, spmatrix]] = None,
        feat_type: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
        is_classification: bool = False,
    ):
        """Fit AutoML to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them. To disable ensembling, set ``ensemble_size==0``.

        # TODO PR1213
        #
        #   `task: Optional[int]` and `is_classification`
        #
        #   `AutoML` tries to identify the task itself with
        #   `sklearn.type_of_target`, leaving little for the subclasses to do.
        #   Except this failes when type_of_target(y) == "multiclass".
        #
        #   "multiclass" be mean either REGRESSION or MULTICLASS_CLASSIFICATION,
        #   and so this is where the subclasses are used to determine which.
        #   However, this could also be deduced from the `is_classification`
        #   paramaeter.
        #
        #   In the future, there is little need for the subclasses of `AutoML`
        #   and no need for the `task` parameter. The extra functionality
        #   provided by `AutoMLClassifier` in predict could be moved to
        #   `AutoSklearnClassifier`, leaving `AutoML` to just produce raw
        #   outputs and simplifying the heirarchy.
        #
        #  `load_models`
        #
        #   This parameter is likely not needed as they are loaded upon demand
        #   throughout `AutoML`.
        #   Creating a @property models that loads models into self.models_ is
        #   not loaded would remove the need for this parameter and simplyify
        #   the verification of `load if self.models_ is None` to one place.
        #
        #   `only_return_configuration_space`
        #
        #   This parameter is indicative of a need to create a seperate method
        #   for this as the functionality of `fit` and what it returns can vary.

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            The target classes.

        task : Optional[int]
            The identifier for the task AutoML is to perform.

        X_test : Optional[{array-like, sparse matrix}, shape (n_samples, n_features)]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : Optional[array-like, shape (n_samples) or (n_samples, n_outputs)]
            Test data target classes. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

        feat_type : Optional[List],
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        dataset_name : Optional[str]
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        only_return_configuration_space: bool = False
            If set to true, fit will only return the configuration space that will
            be used for model search. Otherwise fitting will be performed and an
            ensemble created.

        load_models: bool = True
            If true, this will load the models into memory once complete.

        is_classification: bool = False
            Indicates whether this is a classification task if True or a
            regression task if False.

        Returns
        -------
        self

        """

        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)
        y_test = convert_if_sparse(y_test) if y_test is not None else None

        # Get the task if it doesn't exist
        if task is None:
            y_task = type_of_target(y)
            if not self._supports_task_type(y_task):
                raise ValueError(f"{self.__class__.__name__} does not support"
                                 f" task {y_task}")
            self._task = self._task_type_id(y_task)
        else:
            self._task = task

        # Assign a metric if it doesnt exist
        if self._metric is None:
            self._metric = default_metric_for_task[self._task]

        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        # Create the backend
        self._backend = self._create_backend()

        # By default try to use the TCP logging port or get a new port
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        self._logger = self._get_logger(dataset_name)

        # The first thing we have to do is create the logger to update the backend
        self._backend.setup_logger(self._logger_port)

        if not only_return_configuration_space:
            # If only querying the configuration space, we do not save the start time
            # The start time internally checks for the fit() method to execute only once
            # But this does not apply when only querying the configuration space
            self._backend.save_start_time(self._seed)

        self._stopwatch = StopWatch()

        # Make sure that input is valid
        # Performs Ordinal one hot encoding to the target
        # both for train and test data
        self.InputValidator = InputValidator(
            is_classification=is_classification,
            feat_type=feat_type,
            logger_port=self._logger_port,
        )
        self.InputValidator.fit(X_train=X, y_train=y, X_test=X_test, y_test=y_test)
        X, y = self.InputValidator.transform(X, y)

        if X_test is not None:
            X_test, y_test = self.InputValidator.transform(X_test, y_test)

        X, y = self.subsample_if_too_large(
            X=X,
            y=y,
            logger=self._logger,
            seed=self._seed,
            memory_limit=self._memory_limit,
            task=self._task,
        )

        # Check the re-sampling strategy
        try:
            self._check_resampling_strategy(
                X=X, y=y, task=self._task,
            )
        except Exception as e:
            self._fit_cleanup()
            raise e

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

        self._dataset_name = dataset_name
        self._stopwatch.start_task(self._dataset_name)

        # Take the feature types from the validator
        self._feat_type = self.InputValidator.feature_validator.feat_type

        # Produce debug information to the logfile
        self._logger.debug('Starting to print environment information')
        self._logger.debug('  Python version: %s', sys.version.split('\n'))
        try:
            self._logger.debug(f'\tDistribution: {distro.id()}-{distro.version()}-{distro.name()}')
        except AttributeError:
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
        self._logger.debug('  include: %s', str(self._include))
        self._logger.debug('  exclude: %s', str(self._exclude))
        self._logger.debug('  resampling_strategy: %s', str(self._resampling_strategy))
        self._logger.debug('  resampling_strategy_arguments: %s',
                           str(self._resampling_strategy_arguments))
        self._logger.debug('  n_jobs: %s', str(self._n_jobs))
        self._logger.debug('  multiprocessing_context: %s', str(self._multiprocessing_context))
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
            task=self._task,
            feat_type=self._feat_type,
            dataset_name=dataset_name,
        )

        self._backend._make_internals_directory()
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
            include=self._include,
            exclude=self._exclude,
        )
        if only_return_configuration_space:
            self._fit_cleanup()
            return self.configuration_space

        # == Perform dummy predictions
        # Dummy prediction always have num_run set to 1
        self.num_run += self._do_dummy_prediction(datamanager, num_run=1)

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
                task=self._task,
                metric=self._metric,
                ensemble_size=self._ensemble_size,
                ensemble_nbest=self._ensemble_nbest,
                max_models_on_disc=self._max_models_on_disc,
                seed=self._seed,
                precision=self.precision,
                max_iterations=None,
                read_at_most=np.inf,
                ensemble_memory_limit=self._memory_limit,
                random_state=self._seed,
                logger_port=self._logger_port,
                pynisher_context=self._multiprocessing_context,
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
                start_num_run=self.num_run,
                num_metalearning_cfgs=self._initial_configurations_via_metalearning,
                config_file=configspace_path,
                seed=self._seed,
                metadata_directory=self._metadata_directory,
                metric=self._metric,
                resampling_strategy=self._resampling_strategy,
                resampling_strategy_args=self._resampling_strategy_arguments,
                include=self._include,
                exclude=self._exclude,
                disable_file_output=self._disable_evaluator_output,
                get_smac_object_callback=self._get_smac_object_callback,
                smac_scenario_args=self._smac_scenario_args,
                scoring_functions=self._scoring_functions,
                port=self._logger_port,
                pynisher_context=self._multiprocessing_context,
                ensemble_callback=proc_ensemble,
                trials_callback=self._get_trials_callback
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

        self._logger.info("Starting shutdown...")
        # Wait until the ensemble process is finished to avoid shutting down
        # while the ensemble builder tries to access the data
        if proc_ensemble is not None:
            self.ensemble_performance_history = list(proc_ensemble.history)

            if len(proc_ensemble.futures) > 0:
                # Now we need to wait for the future to return as it cannot be cancelled while it
                # is running: https://stackoverflow.com/a/49203129
                self._logger.info("Ensemble script still running, waiting for it to finish.")
                result = proc_ensemble.futures.pop().result()
                if result:
                    ensemble_history, _, _, _, _ = result
                    self.ensemble_performance_history.extend(ensemble_history)
                self._logger.info("Ensemble script finished, continue shutdown.")

            # save the ensemble performance history file
            if len(self.ensemble_performance_history) > 0:
                pd.DataFrame(self.ensemble_performance_history).to_json(
                        os.path.join(self._backend.internals_directory, 'ensemble_history.json'))

        if load_models:
            self._logger.info("Loading models...")
            self._load_models()
            self._logger.info("Finished loading models...")

        self._fit_cleanup()

        return self

    def _fit_cleanup(self):
        self._logger.info("Closing the dask infrastructure")
        self._close_dask_client()
        self._logger.info("Finished closing the dask infrastructure")

        # Clean up the logger
        self._logger.info("Starting to clean up the logger")
        self._clean_logger()

        # Clean up the backend
        if self._delete_tmp_folder_after_terminate:
            self._backend.context.delete_directories(force=False)
        return

    def _check_resampling_strategy(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        task: int,
    ) -> None:
        """
        This method centralizes the checks for resampling strategies

        Parameters
        ----------
        X: (SUPPORTED_FEAT_TYPES)
            Input features for the given task
        y: (SUPPORTED_TARGET_TYPES)
            Input targets for the given task
        task: (task)
            Integer describing a supported task type, like BINARY_CLASSIFICATION
        """
        is_split_object = isinstance(
            self._resampling_strategy,
            (BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit)
        )

        if self._resampling_strategy not in [
                'holdout',
                'holdout-iterative-fit',
                'cv',
                'cv-iterative-fit',
                'partial-cv',
                'partial-cv-iterative-fit',
        ] and not is_split_object:
            raise ValueError('Illegal resampling strategy: %s' % self._resampling_strategy)

        elif is_split_object:
            TrainEvaluator.check_splitter_resampling_strategy(
                X=X, y=y, task=task,
                groups=self._resampling_strategy_arguments.get('groups', None),
                resampling_strategy=self._resampling_strategy,
            )

        elif self._resampling_strategy in [
            'partial-cv',
            'partial-cv-iterative-fit',
        ] and self._ensemble_size != 0:
            raise ValueError("Resampling strategy %s cannot be used "
                             "together with ensembles." % self._resampling_strategy)

        elif self._resampling_strategy in [
            'partial-cv',
            'cv',
            'cv-iterative-fit',
            'partial-cv-iterative-fit',
        ] and 'folds' not in self._resampling_strategy_arguments:
            self._resampling_strategy_arguments['folds'] = 5

        return

    @staticmethod
    def subsample_if_too_large(
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        logger,
        seed: int,
        memory_limit: int,
        task: int,
    ):
        if memory_limit and isinstance(X, np.ndarray):

            if X.dtype == np.float32:
                multiplier = 4
            elif X.dtype in (np.float64, float):
                multiplier = 8
            elif (
                # In spite of the names, np.float96 and np.float128
                # provide only as much precision as np.longdouble,
                # that is, 80 bits on most x86 machines and 64 bits
                # in standard Windows builds.
                (hasattr(np, 'float128') and X.dtype == np.float128)
                or (hasattr(np, 'float96') and X.dtype == np.float96)
            ):
                multiplier = 16
            else:
                # Just assuming some value - very unlikely
                multiplier = 8
                logger.warning('Unknown dtype for X: %s, assuming it takes 8 bit/number',
                               str(X.dtype))

            megabytes = X.shape[0] * X.shape[1] * multiplier / 1024 / 1024
            if memory_limit <= megabytes * 10 and X.dtype != np.float32:
                cast_to = {
                    8: np.float32,
                    16: np.float64,
                }.get(multiplier, np.float32)
                logger.warning(
                    'Dataset too large for memory limit %dMB, reducing the precision from %s to %s',
                    memory_limit,
                    X.dtype,
                    cast_to,
                )
                X = X.astype(cast_to)

            megabytes = X.shape[0] * X.shape[1] * multiplier / 1024 / 1024
            if memory_limit <= megabytes * 10:
                new_num_samples = int(
                    memory_limit / (10 * X.shape[1] * multiplier / 1024 / 1024)
                )
                logger.warning(
                    'Dataset too large for memory limit %dMB, reducing number of samples from '
                    '%d to %d.',
                    memory_limit,
                    X.shape[0],
                    new_num_samples,
                )
                if task in CLASSIFICATION_TASKS:
                    # Identify if it has unique labels and allow for
                    # stratification, with unique labels in training set
                    values, idxs, counts = np.unique(y, axis=0,
                                                     return_index=True,
                                                     return_counts=True)
                    unique_labels = {
                        idx: value
                        for value, idx, count in zip(values, idxs, counts)
                        if count == 1
                    }

                    # If there are unique labeled elements, remove them and
                    # place them back in later
                    if len(unique_labels) > 0:
                        idxs_of_unique = np.asarray(list(unique_labels.keys()))
                        unique_X = X[idxs_of_unique]
                        unique_y = y[idxs_of_unique]

                        # NOTE optimization
                        #   If this ever turns out to be slow, this actually
                        #   copies the entire array. There might be a better
                        #   solution but it will probably require a lot more
                        #   manual work in how splitting is done.
                        X = np.delete(X, idxs_of_unique, axis=0)
                        y = np.delete(y, idxs_of_unique, axis=0)

                        X, _, y, _ = sklearn.model_selection.train_test_split(
                            X, y,
                            train_size=new_num_samples - len(unique_y),
                            random_state=seed,
                            stratify=y,
                        )

                        X = np.append(X, unique_X, axis=0)
                        y = np.append(y, unique_y, axis=0)

                    # Otherwise we should be able to stratify as normal
                    else:
                        X, _, y, _ = sklearn.model_selection.train_test_split(
                            X, y,
                            train_size=new_num_samples,
                            random_state=seed,
                            stratify=y,
                        )
                elif task in REGRESSION_TASKS:
                    X, _, y, _ = sklearn.model_selection.train_test_split(
                        X, y,
                        train_size=new_num_samples,
                        random_state=seed,
                    )
                else:
                    raise ValueError(task)
        return X, y

    def refit(self, X, y):
        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)

        # Make sure input data is valid
        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError("refit() is only supported after calling fit. Kindly call first "
                             "the estimator fit() method.")
        X, y = self.InputValidator.transform(X, y)

        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # Refit is not applicable when ensemble_size is set to zero.
        if self.ensemble_ is None:
            raise ValueError("Refit can only be called if 'ensemble_size != 0'")

        random_state = check_random_state(self._seed)
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
                            train_indices=np.arange(X.shape[0], dtype=int),
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

    def fit_pipeline(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Union[SUPPORTED_TARGET_TYPES, spmatrix],
        is_classification: bool,
        config: Union[Configuration,  Dict[str, Union[str, float, int]]],
        task: Optional[int] = None,
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[Union[SUPPORTED_TARGET_TYPES, spmatrix]] = None,
        feat_type: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        """ Fits and individual pipeline configuration and returns
        the result to the user.

        The Estimator constraints are honored, for example the resampling
        strategy, or memory constraints, unless directly provided to the method.
        By default, this method supports the same signature as fit(), and any extra
        arguments are redirected to the TAE evaluation function, which allows for
        further customization while building a pipeline.

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            The features used for training
        y: array-like
            The labels used for training
        X_test: Optionalarray-like, shape = (n_samples, n_features)
            If provided, the testing performance will be tracked on this features.
        y_test: array-like
            If provided, the testing performance will be tracked on this labels
        config: Union[Configuration,  Dict[str, Union[str, float, int]]]
            A configuration object used to define the pipeline steps. If a dictionary is passed,
            a configuration is created based on this dictionary.
        dataset_name: Optional[str]
            A string to tag and identify the Auto-Sklearn run
        is_classification: bool
            Whether the task is for classification or regression. This affects
            how the targets are treated
        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        Returns
        -------
        pipeline: Optional[BasePipeline]
            The fitted pipeline. In case of failure while fitting the pipeline,
            a None is returned.
        run_info: RunInFo
            A named tuple that contains the configuration launched
        run_value: RunValue
            A named tuple that contains the result of the run
        """
        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)

        # Get the task if it doesn't exist
        if task is None:
            y_task = type_of_target(y)
            if not self._supports_task_type(y_task):
                raise ValueError(f"{self.__class__.__name__} does not support"
                                 f" task {y_task}")
            self._task = self._task_type_id(y_task)
        else:
            self._task = task

        # Assign a metric if it doesnt exist
        if self._metric is None:
            self._metric = default_metric_for_task[self._task]

        # Get the configuration space
        # This also ensures that the Backend has processed the
        # dataset
        if self.configuration_space is None:
            self.configuration_space = self.fit(
                X=X, y=y,
                dataset_name=dataset_name if dataset_name is not None else self._dataset_name,
                X_test=X_test,
                y_test=y_test,
                feat_type=feat_type,
                only_return_configuration_space=True)

        # We do not want to overwrite existing runs
        self.num_run += 1
        if isinstance(config, dict):
            config = Configuration(self.configuration_space, config)
        config.config_id = self.num_run

        # Prepare missing components to the TAE function call
        if 'include' not in kwargs:
            kwargs['include'] = self._include
        if 'exclude' not in kwargs:
            kwargs['exclude'] = self._exclude
        if 'memory_limit' not in kwargs:
            kwargs['memory_limit'] = self._memory_limit
        if 'resampling_strategy' not in kwargs:
            kwargs['resampling_strategy'] = self._resampling_strategy
        if 'metric' not in kwargs:
            kwargs['metric'] = self._metric
        if 'disable_file_output' not in kwargs:
            kwargs['disable_file_output'] = self._disable_evaluator_output
        if 'pynisher_context' not in kwargs:
            kwargs['pynisher_context'] = self._multiprocessing_context
        if 'stats' not in kwargs:
            scenario_mock = unittest.mock.Mock()
            scenario_mock.wallclock_limit = self._time_for_task
            kwargs['stats'] = Stats(scenario_mock)
        kwargs['stats'].start_timing()

        # Fit a pipeline, which will be stored on disk
        # which we can later load via the backend
        ta = ExecuteTaFuncWithQueue(
            backend=self._backend,
            autosklearn_seed=self._seed,
            abort_on_first_run_crash=False,
            cost_for_crash=get_cost_of_crash(kwargs['metric']),
            port=self._logger_port,
            **kwargs,
            **self._resampling_strategy_arguments
        )

        run_info, run_value = ta.run_wrapper(
            RunInfo(
                config=config,
                instance=None,
                instance_specific=None,
                seed=self._seed,
                cutoff=kwargs.pop('cutoff', self._per_run_time_limit),
                capped=False,
            )
        )

        pipeline = None
        if kwargs['disable_file_output'] or kwargs['resampling_strategy'] == 'test':
            self._logger.warning("File output is disabled. No pipeline can returned")
        elif run_value.status == StatusType.SUCCESS:
            if kwargs['resampling_strategy'] in ('cv', 'cv-iterative-fit'):
                load_function = self._backend.load_cv_model_by_seed_and_id_and_budget
            else:
                load_function = self._backend.load_model_by_seed_and_id_and_budget
            pipeline = load_function(
                seed=self._seed,
                idx=run_info.config.config_id + 1,
                budget=run_info.budget,
            )

        self._clean_logger()

        return pipeline, run_info, run_value

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
        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError("predict() can only be called after performing fit(). Kindly call "
                             "the estimator fit() method first.")
        X = self.InputValidator.feature_validator.transform(X)

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
                model=models[identifier],
                X=X,
                task=self._task,
                batch_size=batch_size
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
        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)

        if self._resampling_strategy in ['partial-cv', 'partial-cv-iterative-fit']:
            raise ValueError('Cannot call fit_ensemble with resampling '
                             'strategy %s.' % self._resampling_strategy)

        if self._logger is None:
            self._logger = self._get_logger(dataset_name)

        # Make sure that input is valid
        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError("fit_ensemble() can only be called after fit. Please call the "
                             "estimator fit() method prior to fit_ensemble().")
        y = self.InputValidator.target_validator.transform(y)

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
            logger_port=self._logger_port,
            pynisher_context=self._multiprocessing_context,
        )
        manager.build_ensemble(self._dask_client)
        future = manager.futures.pop()
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
        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError("score() is only supported after calling fit. Kindly call first "
                             "the estimator fit() method.")
        y = self.InputValidator.target_validator.transform(y)

        # Encode the prediction using the input validator
        # We train autosklearn with a encoded version of y,
        # which is decoded by predict().
        # Above call to validate() encodes the y given for score()
        # Below call encodes the prediction, so we compare in the
        # same representation domain
        prediction = self.InputValidator.target_validator.transform(prediction)

        return calculate_metric(solution=y,
                                prediction=prediction,
                                task_type=self._task,
                                metric=self._metric, )

    def _get_runhistory_models_performance(self):
        metric = self._metric
        data = self.runhistory_.data
        performance_list = []
        for run_key, run_value in data.items():
            if run_value.status != StatusType.SUCCESS:
                # Ignore crashed runs
                continue
            # Alternatively, it is possible to also obtain the start time with
            # ``run_value.starttime``
            endtime = pd.Timestamp(time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(run_value.endtime)))
            val_score = metric._optimum - (metric._sign * run_value.cost)
            train_score = metric._optimum - (metric._sign * run_value.additional_info['train_loss'])
            scores = {
                'Timestamp': endtime,
                'single_best_optimization_score': val_score,
                'single_best_train_score': train_score,
            }
            # Append test-scores, if data for test_loss are available.
            # This is the case, if X_test and y_test where provided.
            if 'test_loss' in run_value.additional_info:
                test_score = metric._optimum - (metric._sign *
                                                run_value.additional_info['test_loss'])
                scores["single_best_test_score"] = test_score

            performance_list.append(scores)
        return pd.DataFrame(performance_list)

    @property
    def performance_over_time_(self):
        individual_performance_frame = self._get_runhistory_models_performance()
        best_values = pd.Series({'single_best_optimization_score': -np.inf,
                                 'single_best_test_score': -np.inf,
                                 'single_best_train_score': -np.inf})
        for idx in individual_performance_frame.index:
            if (
                individual_performance_frame.loc[idx, 'single_best_optimization_score']
                > best_values['single_best_optimization_score']
            ):
                best_values = individual_performance_frame.loc[idx]
            individual_performance_frame.loc[idx] = best_values

        performance_over_time = individual_performance_frame

        if self._ensemble_size != 0:
            ensemble_performance_frame = pd.DataFrame(self.ensemble_performance_history)
            best_values = pd.Series({'ensemble_optimization_score': -np.inf,
                                    'ensemble_test_score': -np.inf})
            for idx in ensemble_performance_frame.index:
                if (
                    ensemble_performance_frame.loc[idx, 'ensemble_optimization_score']
                    > best_values['ensemble_optimization_score']
                ):
                    best_values = ensemble_performance_frame.loc[idx]
                ensemble_performance_frame.loc[idx] = best_values

            performance_over_time = pd.merge(
                ensemble_performance_frame,
                individual_performance_frame,
                on="Timestamp", how='outer'
            ).sort_values('Timestamp').fillna(method='ffill')

        return performance_over_time

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

        metric_mask = dict()
        metric_dict = dict()
        metric_name = []

        for metric in self._scoring_functions:
            metric_name.append(metric.name)
            metric_dict[metric.name] = []
            metric_mask[metric.name] = []

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
            # TODO remove StatusType.RUNNING at some point in the future when the new SMAC 0.13.2
            #  is the new minimum required version!
            elif s in (StatusType.STOP, StatusType.RUNNING):
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

            for metric in self._scoring_functions:
                if metric.name in run_value.additional_info.keys():
                    metric_cost = run_value.additional_info[metric.name]
                    metric_value = metric._optimum - (metric._sign * metric_cost)
                    mask_value = False
                else:
                    metric_value = np.NaN
                    mask_value = True
                metric_dict[metric.name].append(metric_value)
                metric_mask[metric.name].append(mask_value)

        results['mean_test_score'] = np.array(mean_test_score)
        for name in metric_name:
            masked_array = ma.MaskedArray(metric_dict[name],
                                          metric_mask[name])
            results['metric_%s' % name] = masked_array

        results['mean_fit_time'] = np.array(mean_fit_time)
        results['params'] = params
        rank_order = -1 * self._metric._sign * results['mean_test_score']
        results['rank_test_scores'] = scipy.stats.rankdata(rank_order, method='min')
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

    def show_models(self) -> Dict[int, Any]:
        """ Returns a dictionary containing dictionaries of ensemble models.

        Each model in the ensemble can be accessed by giving its ``model_id`` as key.

        A model dictionary contains the following:

        * ``"model_id"`` - The id given to a model by ``autosklearn``.
        * ``"rank"`` - The rank of the model based on it's ``"cost"``.
        * ``"cost"`` - The loss of the model on the validation set.
        * ``"ensemble_weight"`` - The weight given to the model in the ensemble.
        * ``"voting_model"`` - The ``cv_voting_ensemble`` model (for 'cv' resampling).
        * ``"estimators"`` - List of models (dicts) in ``cv_voting_ensemble`` (for 'cv' resampling).
        * ``"data_preprocessor"`` - The preprocessor used on the data.
        * ``"balancing"`` - The balancing used on the data (for classification).
        * ``"feature_preprocessor"`` - The preprocessor for features types.
        * ``"classifier"`` or ``"regressor"`` - The autosklearn wrapped classifier or regressor.
        * ``"sklearn_classifier"`` or ``"sklearn_regressor"`` - The sklearn classifier or regressor.

        **Example**

        .. code-block:: python

            import sklearn.datasets
            import sklearn.metrics
            import autosklearn.regression

            X, y = sklearn.datasets.load_diabetes(return_X_y=True)

            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120
                )
            automl.fit(X_train, y_train, dataset_name='diabetes')

            ensemble_dict = automl.show_models()
            print(ensemble_dict)

        Output:

        .. code-block:: text

            {
                25: {'model_id': 25.0,
                     'rank': 1,
                     'cost': 0.43667876507897496,
                     'ensemble_weight': 0.38,
                     'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing....>,
                     'feature_preprocessor': <autosklearn.pipeline.components....>,
                     'regressor': <autosklearn.pipeline.components.regression....>,
                     'sklearn_regressor': SGDRegressor(alpha=0.0006517033225329654,...)
                    },
                6: {'model_id': 6.0,
                    'rank': 2,
                    'cost': 0.4550418898836528,
                    'ensemble_weight': 0.3,
                    'data_preprocessor': <autosklearn.pipeline.components.data_preprocessing....>,
                    'feature_preprocessor': <autosklearn.pipeline.components....>,
                    'regressor': <autosklearn.pipeline.components.regression....>,
                    'sklearn_regressor': ARDRegression(alpha_1=0.0003701926442639788,...)
                    }...
            }

        Returns
        -------
        Dict(int, Any) : dictionary of length = number of models in the ensemble
            A dictionary of models in the ensemble, where ``model_id`` is the key.

        """

        ensemble_dict = {}

        def has_key(rv, key):
            return rv.additional_info and key in rv.additional_info

        table_dict = {}
        for rkey, rval in self.runhistory_.data.items():
            if has_key(rval, 'num_run'):
                model_id = rval.additional_info['num_run']
                table_dict[model_id] = {
                        'model_id': model_id,
                        'cost': rval.cost
                        }

        # Checking if the dictionary is empty
        if not table_dict:
            raise RuntimeError('No model found. Try increasing \'time_left_for_this_task\'.')

        for i, weight in enumerate(self.ensemble_.weights_):
            (_, model_id, _) = self.ensemble_.identifiers_[i]
            table_dict[model_id]['ensemble_weight'] = weight

        table = pd.DataFrame.from_dict(table_dict, orient='index')

        # Checking which resampling strategy is chosen and selecting the appropriate models
        is_cv = (self._resampling_strategy == "cv")
        models = self.cv_models_ if is_cv else self.models_

        rank = 1  # Initializing rank for the first model
        for (_, model_id, _), model in models.items():
            model_dict = {}  # Declaring model dictionary

            # Inserting model_id, rank, cost and ensemble weight
            model_dict['model_id'] = table.loc[model_id]['model_id'].astype(int)
            model_dict['rank'] = rank
            model_dict['cost'] = table.loc[model_id]['cost']
            model_dict['ensemble_weight'] = table.loc[model_id]['ensemble_weight']
            rank += 1  # Incrementing rank by 1 for the next model

            # The steps in the models pipeline are as follows:
            # 'data_preprocessor': DataPreprocessor,
            # 'balancing': Balancing,
            # 'feature_preprocessor': FeaturePreprocessorChoice,
            # 'classifier'/'regressor': ClassifierChoice/RegressorChoice (autosklearn wrapped model)

            # For 'cv' (cross validation) strategy
            if is_cv:
                # Voting model created by cross validation
                cv_voting_ensemble = model
                model_dict['voting_model'] = cv_voting_ensemble

                # List of models, each trained on one cv fold
                cv_models = []
                for cv_model in cv_voting_ensemble.estimators_:
                    estimator = dict(cv_model.steps)

                    # Adding sklearn model to the model dictionary
                    model_type, autosklearn_wrapped_model = cv_model.steps[-1]
                    estimator[f'sklearn_{model_type}'] = autosklearn_wrapped_model.choice.estimator
                    cv_models.append(estimator)
                model_dict['estimators'] = cv_models

            # For any other strategy
            else:
                steps = dict(model.steps)
                model_dict.update(steps)

                # Adding sklearn model to the model dictionary
                model_type, autosklearn_wrapped_model = model.steps[-1]
                model_dict[f'sklearn_{model_type}'] = autosklearn_wrapped_model.choice.estimator

            # Insterting model_dict in the ensemble dictionary
            ensemble_dict[model_id] = model_dict

        return ensemble_dict

    def _create_search_space(
        self,
        tmp_dir,
        backend,
        datamanager,
        include: Optional[Dict[str, List[str]]] = None,
        exclude: Optional[Dict[str, List[str]]] = None,
    ):
        task_name = 'CreateConfigSpace'

        self._stopwatch.start_task(task_name)
        configspace_path = os.path.join(tmp_dir, 'space.json')
        configuration_space = pipeline.get_configuration_space(
            datamanager.info,
            include=include,
            exclude=exclude,
        )
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
        self.logging_server = None
        self.stop_logging_server = None
        return self.__dict__

    def __del__(self):
        # Clean up the logger
        self._clean_logger()

        self._close_dask_client()


class AutoMLClassifier(AutoML):

    _task_mapping = {
        'multilabel-indicator': MULTILABEL_CLASSIFICATION,
        'multiclass': MULTICLASS_CLASSIFICATION,
        'binary': BINARY_CLASSIFICATION,
    }

    @classmethod
    def _task_type_id(cls, task_type: str) -> int:
        return cls._task_mapping[task_type]

    @classmethod
    def _supports_task_type(cls, task_type: str) -> bool:
        return task_type in cls._task_mapping.keys()

    def fit(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Union[SUPPORTED_TARGET_TYPES, spmatrix],
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[Union[SUPPORTED_TARGET_TYPES, spmatrix]] = None,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ):
        return super().fit(
            X, y,
            X_test=X_test,
            y_test=y_test,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
            is_classification=True,
        )

    def fit_pipeline(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Union[SUPPORTED_TARGET_TYPES, spmatrix],
        config: Union[Configuration,  Dict[str, Union[str, float, int]]],
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[Union[SUPPORTED_TARGET_TYPES, spmatrix]] = None,
        feat_type: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        return super().fit_pipeline(
            X=X, y=y,
            X_test=X_test, y_test=y_test,
            dataset_name=dataset_name,
            config=config,
            is_classification=True,
            feat_type=feat_type,
            **kwargs,
        )

    def predict(self, X, batch_size=None, n_jobs=1):
        predicted_probabilities = super().predict(X, batch_size=batch_size,
                                                  n_jobs=n_jobs)

        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError("predict() is only supported after calling fit. Kindly call first "
                             "the estimator fit() method.")
        if self.InputValidator.target_validator.is_single_column_target():
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
        else:
            predicted_indexes = (predicted_probabilities > 0.5).astype(int)

        return self.InputValidator.target_validator.inverse_transform(predicted_indexes)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)


class AutoMLRegressor(AutoML):

    _task_mapping = {
        'continuous-multioutput': MULTIOUTPUT_REGRESSION,
        'continuous': REGRESSION,
        'multiclass': REGRESSION,
    }

    @classmethod
    def _task_type_id(cls, task_type: str) -> int:
        return cls._task_mapping[task_type]

    @classmethod
    def _supports_task_type(cls, task_type: str) -> bool:
        return task_type in cls._task_mapping.keys()

    def fit(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Union[SUPPORTED_TARGET_TYPES, spmatrix],
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[Union[SUPPORTED_TARGET_TYPES, spmatrix]] = None,
        feat_type: Optional[List[bool]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ):
        return super().fit(
            X, y,
            X_test=X_test,
            y_test=y_test,
            feat_type=feat_type,
            dataset_name=dataset_name,
            only_return_configuration_space=only_return_configuration_space,
            load_models=load_models,
            is_classification=False,
        )

    def fit_pipeline(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Union[SUPPORTED_TARGET_TYPES, spmatrix],
        config: Union[Configuration,  Dict[str, Union[str, float, int]]],
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[Union[SUPPORTED_TARGET_TYPES, spmatrix]] = None,
        feat_type: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        return super().fit_pipeline(
            X=X, y=y,
            X_test=X_test, y_test=y_test,
            config=config,
            feat_type=feat_type,
            dataset_name=dataset_name,
            is_classification=False,
            **kwargs,
        )
