from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import copy
import io
import itertools
import json
import logging.handlers
import multiprocessing
import os
import platform
import sys
import time
import types
import uuid
import warnings

import distro
import joblib
import numpy as np
import numpy.ma as ma
import pandas as pd
import pkg_resources
import scipy.stats
import sklearn.utils
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json
from dask.distributed import Client
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics._classification import type_of_target
from sklearn.model_selection._split import (
    BaseCrossValidator,
    BaseShuffleSplit,
    _RepeatedSplits,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from smac.callbacks import IncorporateRunResultCallback
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.stats.stats import Stats
from smac.tae import StatusType
from typing_extensions import Literal

from autosklearn.automl_common.common.utils.backend import Backend, create
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    CLASSIFICATION_TASKS,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    REGRESSION_TASKS,
)
from autosklearn.data.validation import (
    SUPPORTED_FEAT_TYPES,
    SUPPORTED_TARGET_TYPES,
    InputValidator,
    convert_if_sparse,
)
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.ensemble_building import EnsembleBuilderManager
from autosklearn.ensembles.abstract_ensemble import (
    AbstractEnsemble,
    AbstractMultiObjectiveEnsemble,
)
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.ensembles.singlebest_ensemble import SingleBestFromRunhistory
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.evaluation.abstract_evaluator import _fit_and_suppress_warnings
from autosklearn.evaluation.train_evaluator import TrainEvaluator, _fit_with_budget
from autosklearn.metrics import (
    Scorer,
    _validate_metrics,
    compute_single_metric,
    default_metric_for_task,
)
from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding import (
    OHEChoice,
)
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense import (
    CoalescenseChoice,
)
from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
from autosklearn.pipeline.components.feature_preprocessing import (
    FeaturePreprocessorChoice,
)
from autosklearn.pipeline.components.regression import RegressorChoice
from autosklearn.smbo import AutoMLSMBO
from autosklearn.util import RE_PATTERN, pipeline
from autosklearn.util.dask import Dask, LocalDask, UserDask
from autosklearn.util.data import (
    DatasetCompressionSpec,
    default_dataset_compression_arg,
    reduce_dataset_size_if_too_large,
    supported_precision_reductions,
    validate_dataset_compression_arg,
)
from autosklearn.util.logging_ import (
    PicklableClientLogger,
    get_named_client_logger,
    setup_logger,
    start_log_server,
    warnings_to,
)
from autosklearn.util.parallel import preload_modules
from autosklearn.util.smac_wrap import SMACCallback, SmacRunCallback
from autosklearn.util.stopwatch import StopWatch

import unittest.mock


def _model_predict(
    model: Any,
    X: SUPPORTED_FEAT_TYPES,
    task: int,
    batch_size: Optional[int] = None,
    logger: Optional[PicklableClientLogger] = None,
) -> np.ndarray:
    """Generates the predictions from a model.

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

            if batch_size is not None and hasattr(model, "batch_size"):
                prediction = predict_func(X_, batch_size=batch_size)
            else:
                prediction = predict_func(X_)

    # Check that probability values lie between 0 and 1.
    if task in CLASSIFICATION_TASKS:
        assert (prediction >= 0).all() and (
            prediction <= 1
        ).all(), f"For {model}, prediction probability not within [0, 1]!"

    assert (
        prediction.shape[0] == X_.shape[0]
    ), f"Prediction shape {model} is {prediction.shape} while X_.shape is {X_.shape}"

    return prediction


class AutoML(BaseEstimator):
    """Base class for handling the AutoML procedure"""

    def __init__(
        self,
        time_left_for_this_task: int,
        per_run_time_limit: int,
        temporary_directory: Optional[str] = None,
        delete_tmp_folder_after_terminate: bool = True,
        initial_configurations_via_metalearning: int = 25,
        ensemble_class: Type[AbstractEnsemble] | None = EnsembleSelection,
        ensemble_kwargs: Dict[str, Any] | None = None,
        ensemble_nbest: int = 1,
        max_models_on_disc: int = 1,
        seed: int = 1,
        memory_limit: int = 3072,
        metadata_directory: Optional[str] = None,
        include: Optional[dict[str, list[str]]] = None,
        exclude: Optional[dict[str, list[str]]] = None,
        resampling_strategy: str | Any = "holdout-iterative-fit",
        resampling_strategy_arguments: Mapping[str, Any] = None,
        n_jobs: Optional[int] = None,
        dask_client: Optional[Client] = None,
        precision: Literal[16, 32, 64] = 32,
        disable_evaluator_output: bool | Iterable[str] = False,
        get_smac_object_callback: Optional[Callable] = None,
        smac_scenario_args: Optional[Mapping] = None,
        logging_config: Optional[Mapping] = None,
        metrics: Sequence[Scorer] | None = None,
        scoring_functions: Optional[list[Scorer]] = None,
        get_trials_callback: SMACCallback | None = None,
        dataset_compression: bool | Mapping[str, Any] = True,
        allow_string_features: bool = True,
    ):
        super().__init__()

        if isinstance(disable_evaluator_output, Iterable):
            disable_evaluator_output = list(disable_evaluator_output)  # Incase iterator
            allowed = set(["model", "cv_model", "y_optimization", "y_test"])
            unknown = allowed - set(disable_evaluator_output)
            if any(unknown):
                raise ValueError(
                    f"Unknown arg {unknown} for '_disable_evaluator_output',"
                    f" must be one of {allowed}"
                )

        # Validate dataset_compression and set its values
        self._dataset_compression: Optional[DatasetCompressionSpec]
        if isinstance(dataset_compression, bool):
            if dataset_compression is True:
                self._dataset_compression = default_dataset_compression_arg
            else:
                self._dataset_compression = None
        else:
            self._dataset_compression = validate_dataset_compression_arg(
                dataset_compression,
                memory_limit=memory_limit,
            )

        # If we got something callable for `get_trials_callback`, wrap it so SMAC
        # will accept it.
        if (
            get_trials_callback is not None
            and callable(get_trials_callback)
            and not isinstance(get_trials_callback, IncorporateRunResultCallback)
        ):
            get_trials_callback = SmacRunCallback(get_trials_callback)

        self._delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self._time_for_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._metrics = metrics
        self._ensemble_class = ensemble_class
        self._ensemble_kwargs = ensemble_kwargs
        self._ensemble_nbest = ensemble_nbest
        self._max_models_on_disc = max_models_on_disc
        self._seed = seed
        self._memory_limit = memory_limit
        self._metadata_directory = metadata_directory
        self._include = include
        self._exclude = exclude
        self._resampling_strategy = resampling_strategy
        self._disable_evaluator_output = disable_evaluator_output
        self._get_smac_object_callback = get_smac_object_callback
        self._get_trials_callback = get_trials_callback
        self._smac_scenario_args = smac_scenario_args
        self.logging_config = logging_config
        self.precision = precision
        self.allow_string_features = allow_string_features
        self._initial_configurations_via_metalearning = (
            initial_configurations_via_metalearning
        )
        self._n_jobs = n_jobs

        self._scoring_functions = scoring_functions or []
        self._resampling_strategy_arguments = resampling_strategy_arguments or {}
        self._multiprocessing_context = "forkserver"

        # Single core, local runs should use fork to prevent the __main__ requirements
        # in examples. Nevertheless, multi-process runs have spawn as requirement to
        # reduce the possibility of a deadlock
        self._dask: Dask
        if dask_client is not None:
            self._dask = UserDask(client=dask_client)
        else:
            self._dask = LocalDask(n_jobs=n_jobs)
            if n_jobs == 1:
                self._multiprocessing_context = "fork"

        # Create the backend
        self._backend: Backend = create(
            temporary_directory=temporary_directory,
            output_directory=None,
            prefix="auto-sklearn",
            delete_output_folder_after_terminate=delete_tmp_folder_after_terminate,
        )

        self._data_memory_limit = None  # TODO: dead variable? Always None
        self._datamanager = None
        self._dataset_name = None
        self._feat_type = None
        self._logger: Optional[PicklableClientLogger] = None
        self._task = None
        self._label_num = None
        self._parser = None
        self._can_predict = False
        self._read_at_most = None
        self._max_ensemble_build_iterations = None
        self.models_: Optional[dict] = None
        self.cv_models_: Optional[dict] = None
        self.ensemble_ = None
        self.InputValidator: Optional[InputValidator] = None
        self.configuration_space = None

        # The ensemble performance history through time
        self._stopwatch = StopWatch()
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
        self.ensemble_performance_history = []

        # Num_run tell us how many runs have been launched. It can be seen as an
        # identifier for each configuration saved to disk
        self.num_run = 0
        self.fitted = False

    def _get_logger(self, name: str) -> PicklableClientLogger:
        logger_name = "AutoML(%d):%s" % (self._seed, name)

        # Setup the configuration for the logger
        # This is gonna be honored by the server
        # Which is created below
        setup_logger(
            filename="%s.log" % str(logger_name),
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
        port = context.Value("l")  # be safe by using a long
        port.value = -1

        self.logging_server = context.Process(
            target=start_log_server,
            kwargs=dict(
                host="localhost",
                logname=logger_name,
                event=self.stop_logging_server,
                port=port,
                filename="%s.log" % str(logger_name),
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
            host="localhost",
            port=self._logger_port,
        )

    def _clean_logger(self) -> None:
        if not hasattr(self, "stop_logging_server") or self.stop_logging_server is None:
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

    def _do_dummy_prediction(self) -> None:
        # When using partial-cv it makes no sense to do dummy predictions
        if self._resampling_strategy in ["partial-cv", "partial-cv-iterative-fit"]:
            return

        if self._metrics is None:
            raise ValueError("Metric/Metrics was/were not set")

        # Dummy prediction always have num_run set to 1
        dummy_run_num = 1

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
        ta = ExecuteTaFuncWithQueue(
            backend=self._backend,
            autosklearn_seed=self._seed,
            multi_objectives=[metric.name for metric in self._metrics],
            resampling_strategy=self._resampling_strategy,
            initial_num_run=dummy_run_num,
            stats=stats,
            metrics=self._metrics,
            memory_limit=memory_limit,
            disable_file_output=self._disable_evaluator_output,
            abort_on_first_run_crash=False,
            cost_for_crash=get_cost_of_crash(self._metrics),
            port=self._logger_port,
            pynisher_context=self._multiprocessing_context,
            **self._resampling_strategy_arguments,
        )

        status, cost, runtime, additional_info = ta.run(
            config=dummy_run_num,
            cutoff=self._time_for_task,
        )
        if status == StatusType.SUCCESS:
            self._logger.info("Finished creating dummy predictions.")

        # Fail if dummy prediction fails.
        else:
            if additional_info.get("exitcode") == -6:
                msg = (
                    f"Dummy prediction failed with run state {status}."
                    " The error suggests that the provided memory limits are too tight."
                    " Please increase the 'memory_limit' and try again. If this does"
                    " not solve your problem, please open an issue and paste the"
                    f" additional output. Additional output: {additional_info}"
                )
            else:
                msg = (
                    f" Dummy prediction failed with run state {status} and"
                    f" additional output: {additional_info}.",
                )

            self._logger.error(msg)
            raise ValueError(msg)

        return

    @classmethod
    def _task_type_id(cls, task_type: str) -> int:
        raise NotImplementedError

    @classmethod
    def _supports_task_type(cls, task_type: str) -> bool:
        raise NotImplementedError

    def fit(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        task: Optional[int] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
        feat_type: Optional[list[str]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
        is_classification: bool = False,
    ):
        """Fit AutoML to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them.

        # TODO PR1213
        #
        #   `task: Optional[int]` and `is_classification`
        #
        #   `AutoML` tries to identify the task itself with `sklearn.type_of_target`,
        #   leaving little for the subclasses to do.
        #   Except this failes when type_of_target(y) == "multiclass".
        #
        #   "multiclass" be mean either REGRESSION or MULTICLASS_CLASSIFICATION,
        #   and so this is where the subclasses are used to determine which.
        #   However, this could also be deduced from the `is_classification`
        #   parameter.
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

        feat_type : Optional[list],
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
        if (X_test is not None) ^ (y_test is not None):
            raise ValueError("Must provide both X_test and y_test together")

        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)
        y_test = convert_if_sparse(y_test) if y_test is not None else None

        # Get the task if it doesn't exist
        if task is None:
            y_task = type_of_target(y)
            if not self._supports_task_type(y_task):
                raise ValueError(
                    f"{self.__class__.__name__} does not support" f" task {y_task}"
                )
            self._task = self._task_type_id(y_task)
        else:
            self._task = task

        # Assign a metric if it doesnt exist
        if self._metrics is None:
            self._metrics = [default_metric_for_task[self._task]]
        _validate_metrics(self._metrics, self._scoring_functions)

        if dataset_name is None:
            dataset_name = str(uuid.uuid1(clock_seq=os.getpid()))

        # By default try to use the TCP logging port or get a new port
        self._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        # Once we start the logging server, it starts in a new process
        # If an error occurs then we want to make sure that we exit cleanly
        # and shut it down, else it might hang
        # https://github.com/automl/auto-sklearn/issues/1480
        try:
            self._logger = self._get_logger(dataset_name)

            # The first thing we have to do is create the logger to update the backend
            self._backend.setup_logger(self._logger_port)

            if not only_return_configuration_space:
                # If only querying the configuration space, we do not save the start
                # time The start time internally checks for the fit() method to execute
                # only once but this does not apply when only querying the configuration
                # space
                self._backend.save_start_time(self._seed)

            self._stopwatch = StopWatch()

            # Make sure that input is valid
            # Performs Ordinal one hot encoding to the target
            # both for train and test data
            self.InputValidator = InputValidator(
                is_classification=is_classification,
                feat_type=feat_type,
                logger_port=self._logger_port,
                allow_string_features=self.allow_string_features,
            )
            self.InputValidator.fit(X_train=X, y_train=y, X_test=X_test, y_test=y_test)
            X, y = self.InputValidator.transform(X, y)

            if X_test is not None and y_test is not None:
                X_test, y_test = self.InputValidator.transform(X_test, y_test)

            # We don't support size reduction on pandas type object yet
            if (
                self._dataset_compression is not None
                and not isinstance(X, pd.DataFrame)
                and not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame))
            ):
                methods = self._dataset_compression["methods"]
                memory_allocation = self._dataset_compression["memory_allocation"]

                # Remove precision reduction if we can't perform it
                if (
                    "precision" in methods
                    and X.dtype not in supported_precision_reductions
                ):
                    methods = [method for method in methods if method != "precision"]

                with warnings_to(self._logger):
                    X, y = reduce_dataset_size_if_too_large(
                        X=X,
                        y=y,
                        memory_limit=self._memory_limit,
                        is_classification=is_classification,
                        random_state=self._seed,
                        operations=methods,
                        memory_allocation=memory_allocation,
                    )

            # Check the re-sampling strategy
            self._check_resampling_strategy(
                X=X,
                y=y,
                task=self._task,
            )

            # Reset learnt stuff
            self.models_ = None
            self.cv_models_ = None
            self.ensemble_ = None

            # The metric must exist as of this point
            # It can be provided in the constructor, or automatically
            # defined in the estimator fit call
            if isinstance(self._metrics, Sequence):
                for entry in self._metrics:
                    if not isinstance(entry, Scorer):
                        raise ValueError(
                            f"Metric {entry} must be instance of"
                            " autosklearn.metrics.Scorer."
                        )
            else:
                raise ValueError(
                    "Metric must be a sequence of instances of "
                    "autosklearn.metrics.Scorer."
                )

            self._dataset_name = dataset_name
            self._stopwatch.start(self._dataset_name)

            # Take the feature types from the validator
            self._feat_type = self.InputValidator.feature_validator.feat_type

            self._log_fit_setup()

            # == Pickle the data manager to speed up loading
            with self._stopwatch.time("Save Datamanager"):
                datamanager = XYDataManager(
                    X,
                    y,
                    X_test=X_test,
                    y_test=y_test,
                    task=self._task,
                    feat_type=self._feat_type,
                    dataset_name=dataset_name,
                )

                self._backend._make_internals_directory()
                self._label_num = datamanager.info["label_num"]

                self._backend.save_datamanager(datamanager)

            # = Create a searchspace
            # Do this before One Hot Encoding to make sure that it creates a
            # search space for a dense classifier even if one hot encoding would
            # make it sparse (tradeoff; if one hot encoding would make it sparse,
            #  densifier and truncatedSVD would probably lead to a MemoryError,
            # like this we can't use some of the preprocessing methods in case
            # the data became sparse)
            with self._stopwatch.time("Create Search space"):
                self.configuration_space, configspace_path = self._create_search_space(
                    self._backend.temporary_directory,
                    self._backend,
                    datamanager,
                    include=self._include,
                    exclude=self._exclude,
                )

            if only_return_configuration_space:
                return self.configuration_space

            # == Perform dummy predictions
            with self._stopwatch.time("Dummy predictions"):
                self.num_run += 1
                self._do_dummy_prediction()

            # == RUN ensemble builder
            # Do this before calculating the meta-features to make sure that the
            # dummy predictions are actually included in the ensemble even if
            # calculating the meta-features takes very long
            with self._stopwatch.time("Run Ensemble Builder"):

                elapsed_time = self._stopwatch.time_since(self._dataset_name, "start")

                time_left_for_ensembles = max(0, self._time_for_task - elapsed_time)
                proc_ensemble = None
                if time_left_for_ensembles <= 0:
                    # Fit only raises error when an ensemble class is given but
                    # time_left_for_ensembles is zero.
                    if self._ensemble_class is not None:
                        raise ValueError(
                            "Not starting ensemble builder because there "
                            "is no time left. Try increasing the value "
                            "of time_left_for_this_task."
                        )
                elif self._ensemble_class is None:
                    self._logger.info(
                        "No ensemble buildin because no ensemble class was given."
                    )
                else:
                    self._logger.info(
                        "Start Ensemble with %5.2fsec time left"
                        % time_left_for_ensembles
                    )

                    proc_ensemble = EnsembleBuilderManager(
                        start_time=time.time(),
                        time_left_for_ensembles=time_left_for_ensembles,
                        backend=copy.deepcopy(self._backend),
                        dataset_name=dataset_name,
                        task=self._task,
                        metrics=self._metrics,
                        ensemble_class=self._ensemble_class,
                        ensemble_kwargs=self._ensemble_kwargs,
                        ensemble_nbest=self._ensemble_nbest,
                        max_models_on_disc=self._max_models_on_disc,
                        seed=self._seed,
                        precision=self.precision,
                        max_iterations=self._max_ensemble_build_iterations,
                        read_at_most=self._read_at_most,
                        memory_limit=self._memory_limit,
                        random_state=self._seed,
                        logger_port=self._logger_port,
                        pynisher_context=self._multiprocessing_context,
                    )

            # kill the datamanager as it will be re-loaded anyways from sub processes
            try:
                del self._datamanager
            except Exception:
                pass

            # => RUN SMAC
            with self._stopwatch.time("Run SMAC"):
                elapsed_time = self._stopwatch.time_since(self._dataset_name, "start")
                time_left = self._time_for_task - elapsed_time

                if self._logger:
                    self._logger.info("Start SMAC with %5.2fsec time left" % time_left)
                if time_left <= 0:
                    self._logger.warning(
                        "Not starting SMAC because there is no time left."
                    )
                    _proc_smac = None
                    self._budget_type = None
                else:
                    if (
                        self._per_run_time_limit is None
                        or self._per_run_time_limit > time_left
                    ):
                        self._logger.warning(
                            "Time limit for a single run is higher than total time "
                            "limit. Capping the limit for a single run to the total "
                            "time given to SMAC (%f)" % time_left
                        )
                        per_run_time_limit = time_left
                    else:
                        per_run_time_limit = self._per_run_time_limit

                    # At least 2 models are created for the ensemble process
                    num_models = time_left // per_run_time_limit
                    if num_models < 2:
                        per_run_time_limit = time_left // 2
                        self._logger.warning(
                            "Capping the per_run_time_limit to {} to have "
                            "time for a least 2 models in each process.".format(
                                per_run_time_limit
                            )
                        )

                    n_meta_configs = self._initial_configurations_via_metalearning
                    with self._dask as dask_client:
                        resamp_args = self._resampling_strategy_arguments
                        _proc_smac = AutoMLSMBO(
                            config_space=self.configuration_space,
                            dataset_name=self._dataset_name,
                            backend=self._backend,
                            total_walltime_limit=time_left,
                            func_eval_time_limit=per_run_time_limit,
                            memory_limit=self._memory_limit,
                            data_memory_limit=self._data_memory_limit,
                            stopwatch=self._stopwatch,
                            n_jobs=self._n_jobs,
                            dask_client=dask_client,
                            start_num_run=self.num_run,
                            num_metalearning_cfgs=n_meta_configs,
                            config_file=configspace_path,
                            seed=self._seed,
                            metadata_directory=self._metadata_directory,
                            metrics=self._metrics,
                            resampling_strategy=self._resampling_strategy,
                            resampling_strategy_args=resamp_args,
                            include=self._include,
                            exclude=self._exclude,
                            disable_file_output=self._disable_evaluator_output,
                            get_smac_object_callback=self._get_smac_object_callback,
                            smac_scenario_args=self._smac_scenario_args,
                            scoring_functions=self._scoring_functions,
                            port=self._logger_port,
                            pynisher_context=self._multiprocessing_context,
                            ensemble_callback=proc_ensemble,
                            trials_callback=self._get_trials_callback,
                        )

                        (
                            self.runhistory_,
                            self.trajectory_,
                            self._budget_type,
                        ) = _proc_smac.run_smbo()

                        trajectory_filename = os.path.join(
                            self._backend.get_smac_output_directory_for_run(self._seed),
                            "trajectory.json",
                        )
                        saveable_trajectory = [
                            list(entry[:2])
                            + [entry[2].get_dictionary()]
                            + list(entry[3:])
                            for entry in self.trajectory_
                        ]
                        with open(trajectory_filename, "w") as fh:
                            json.dump(saveable_trajectory, fh)

                        self._logger.info("Starting shutdown...")
                        # Wait until the ensemble process is finished to avoid shutting
                        # down while the ensemble builder tries to access the data
                        if proc_ensemble is not None:
                            self.ensemble_performance_history = list(
                                proc_ensemble.history
                            )

                            if len(proc_ensemble.futures) > 0:
                                # Now we wait for the future to return as it cannot be
                                # cancelled while it is running
                                # * https://stackoverflow.com/a/49203129
                                self._logger.info(
                                    "Ensemble script still running,"
                                    " waiting for it to finish."
                                )
                                result = proc_ensemble.futures.pop().result()

                            if result:
                                ensemble_history, _ = result
                                self.ensemble_performance_history.extend(
                                    ensemble_history
                                )

                            self._logger.info(
                                "Ensemble script finished, continue shutdown."
                            )

                # save the ensemble performance history file
                if len(self.ensemble_performance_history) > 0:
                    pd.DataFrame(self.ensemble_performance_history).to_json(
                        os.path.join(
                            self._backend.internals_directory, "ensemble_history.json"
                        )
                    )

            if load_models:
                self._logger.info("Loading models...")
                self._load_models()
                self._logger.info("Finished loading models...")

        # The whole logic above from where we begin the logging server is capture
        # in a try: finally: so that if something goes wrong, we at least close
        # down the logging server, preventing it from hanging and not closing
        # until ctrl+c is pressed
        except Exception as e:
            # This will be called before the _fit_cleanup
            self._logger.exception(e)
            raise e
        finally:
            self._fit_cleanup()

        self.fitted = True

        return self

    def _log_fit_setup(self) -> None:
        # Produce debug information to the logfile
        self._logger.debug("Starting to print environment information")
        self._logger.debug("  Python version: %s", sys.version.split("\n"))
        try:
            self._logger.debug(
                f"\tDistribution: {distro.id()}-{distro.version()}-{distro.name()}"
            )
        except AttributeError:
            pass

        self._logger.debug("  System: %s", platform.system())
        self._logger.debug("  Machine: %s", platform.machine())
        self._logger.debug("  Platform: %s", platform.platform())
        # UNAME appears to leak sensible information
        # self._logger.debug('  uname: %s', platform.uname())
        self._logger.debug("  Version: %s", platform.version())
        self._logger.debug("  Mac version: %s", platform.mac_ver())
        requirements = pkg_resources.resource_string("autosklearn", "requirements.txt")
        requirements = requirements.decode("utf-8")
        requirements = [requirement for requirement in requirements.split("\n")]
        for requirement in requirements:
            if not requirement:
                continue
            match = RE_PATTERN.match(requirement)
            if match:
                name = match.group("name")
                module_dist = pkg_resources.get_distribution(name)
                self._logger.debug("  %s", module_dist)
            else:
                raise ValueError("Unable to read requirement: %s" % requirement)

        self._logger.debug("Done printing environment information")
        self._logger.debug("Starting to print arguments to auto-sklearn")
        self._logger.debug("  tmp_folder: %s", self._backend.temporary_directory)
        self._logger.debug("   time_left_for_this_task: %f", self._time_for_task)
        self._logger.debug("  per_run_time_limit: %f", self._per_run_time_limit)
        self._logger.debug(
            "  initial_configurations_via_metalearning: %d",
            self._initial_configurations_via_metalearning,
        )
        self._logger.debug("  ensemble_class: %s", self._ensemble_class)
        self._logger.debug("  ensemble_kwargs: %s", self._ensemble_kwargs)
        self._logger.debug("  ensemble_nbest: %f", self._ensemble_nbest)
        self._logger.debug("  max_models_on_disc: %s", str(self._max_models_on_disc))
        self._logger.debug("  seed: %d", self._seed)
        self._logger.debug("  memory_limit: %s", str(self._memory_limit))
        self._logger.debug("  metadata_directory: %s", self._metadata_directory)
        self._logger.debug("  include: %s", str(self._include))
        self._logger.debug("  exclude: %s", str(self._exclude))
        self._logger.debug("  resampling_strategy: %s", str(self._resampling_strategy))
        self._logger.debug(
            "  resampling_strategy_arguments: %s",
            str(self._resampling_strategy_arguments),
        )
        self._logger.debug("  n_jobs: %s", str(self._n_jobs))
        self._logger.debug(
            "  multiprocessing_context: %s", str(self._multiprocessing_context)
        )
        self._logger.debug("  dask_client: %s", str(self._dask))
        self._logger.debug("  precision: %s", str(self.precision))
        self._logger.debug(
            "  disable_evaluator_output: %s", str(self._disable_evaluator_output)
        )
        self._logger.debug(
            "  get_smac_objective_callback: %s", str(self._get_smac_object_callback)
        )
        self._logger.debug("  smac_scenario_args: %s", str(self._smac_scenario_args))
        self._logger.debug("  logging_config: %s", str(self.logging_config))
        if len(self._metrics) == 1:
            self._logger.debug("  metric: %s", str(self._metrics[0]))
        else:
            self._logger.debug("  metrics: %s", str(self._metrics))
        self._logger.debug("Done printing arguments to auto-sklearn")
        self._logger.debug("Starting to print available components")
        for choice in (
            ClassifierChoice,
            RegressorChoice,
            FeaturePreprocessorChoice,
            OHEChoice,
            RescalingChoice,
            CoalescenseChoice,
        ):
            self._logger.debug(
                "%s: %s",
                choice.__name__,
                choice.get_components(),
            )
        self._logger.debug("Done printing available components")

    def __sklearn_is_fitted__(self) -> bool:
        return self.fitted

    def _fit_cleanup(self) -> None:
        self._logger.info("Closing the dask infrastructure")
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
            (BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit),
        )

        if (
            self._resampling_strategy
            not in [
                "holdout",
                "holdout-iterative-fit",
                "cv",
                "cv-iterative-fit",
                "partial-cv",
                "partial-cv-iterative-fit",
            ]
            and not is_split_object
        ):
            raise ValueError(
                "Illegal resampling strategy: %s" % self._resampling_strategy
            )

        elif is_split_object:
            TrainEvaluator.check_splitter_resampling_strategy(
                X=X,
                y=y,
                task=task,
                groups=self._resampling_strategy_arguments.get("groups", None),
                resampling_strategy=self._resampling_strategy,
            )

        elif (
            self._resampling_strategy
            in [
                "partial-cv",
                "partial-cv-iterative-fit",
            ]
            and self._ensemble_class is not None
        ):
            raise ValueError(
                "Resampling strategy %s cannot be used "
                "together with ensembles." % self._resampling_strategy
            )

        elif (
            self._resampling_strategy
            in [
                "partial-cv",
                "cv",
                "cv-iterative-fit",
                "partial-cv-iterative-fit",
            ]
            and "folds" not in self._resampling_strategy_arguments
        ):
            self._resampling_strategy_arguments["folds"] = 5

        return

    def refit(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        max_reshuffles: int = 10,
    ) -> AutoML:
        """Refit the models to a new given set of data

        Parameters
        ----------
        X : SUPPORTED_FEAT_TYPES
            The data to dit to

        y : SUPPORTED_TARGET_TYPES
            The targets to fit to

        max_reshuffles : int = 10
            How many times to try reshuffle the data. If fitting fails, shuffle the
            data. This can alleviate the problem in algorithms that depend on the
            ordering of the data.

        Returns
        -------
        AutoML
            Self
        """
        check_is_fitted(self)
        y = convert_if_sparse(y)  # AutoSklearn does not handle sparse y for now

        # Make sure input data is valid
        X, y = self.InputValidator.transform(X, y)

        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # Refit is not applicable when no ensemble class is provided
        if self.ensemble_ is None:
            raise ValueError("Refit can only be called if an ensemble class is given")

        random_state = check_random_state(self._seed)

        for identifier, model in self.models_.items():
            for i in range(max_reshuffles):
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

                    if i == (max_reshuffles - 1):
                        raise e

        self._can_predict = True
        return self

    def fit_pipeline(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES | spmatrix,
        is_classification: bool,
        config: Configuration | dict[str, str | float | int],
        task: Optional[int] = None,
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES | spmatrix] = None,
        feat_type: Optional[list[str]] = None,
        **kwargs: dict,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        """Fits and individual pipeline configuration and returns
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
        config: Configuration |  dict[str, str | float | int]
            A configuration object used to define the pipeline steps.
            If a dict is passed, a configuration is created based on this dict.
        dataset_name: Optional[str]
            A string to tag and identify the Auto-Sklearn run
        is_classification: bool
            Whether the task is for classification or regression. This affects
            how the targets are treated
        feat_type : List, optional (default=None)
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
                raise ValueError(
                    f"{self.__class__.__name__} does not support" f" task {y_task}"
                )
            self._task = self._task_type_id(y_task)
        else:
            self._task = task

        # Assign a metric if it doesnt exist
        if self._metrics is None:
            self._metrics = [default_metric_for_task[self._task]]

        # Get the configuration space
        # This also ensures that the Backend has processed the
        # dataset
        if self.configuration_space is None:
            self.configuration_space = self.fit(
                X=X,
                y=y,
                dataset_name=dataset_name
                if dataset_name is not None
                else self._dataset_name,
                X_test=X_test,
                y_test=y_test,
                feat_type=feat_type,
                only_return_configuration_space=True,
            )

        # We do not want to overwrite existing runs
        self.num_run += 1
        if isinstance(config, dict):
            config = Configuration(self.configuration_space, config)
        config.config_id = self.num_run

        # Prepare missing components to the TAE function call
        if "include" not in kwargs:
            kwargs["include"] = self._include
        if "exclude" not in kwargs:
            kwargs["exclude"] = self._exclude
        if "memory_limit" not in kwargs:
            kwargs["memory_limit"] = self._memory_limit
        if "resampling_strategy" not in kwargs:
            kwargs["resampling_strategy"] = self._resampling_strategy
        if "metrics" not in kwargs:
            if "metric" in kwargs:
                kwargs["metrics"] = kwargs["metric"]
                del kwargs["metric"]
            else:
                kwargs["metrics"] = self._metrics
        if not isinstance(kwargs["metrics"], Sequence):
            kwargs["metrics"] = [kwargs["metrics"]]
        if "scoring_functions" not in kwargs:
            kwargs["scoring_functions"] = self._scoring_functions
        if "disable_file_output" not in kwargs:
            kwargs["disable_file_output"] = self._disable_evaluator_output
        if "pynisher_context" not in kwargs:
            kwargs["pynisher_context"] = self._multiprocessing_context
        if "stats" not in kwargs:
            scenario_mock = unittest.mock.Mock()
            scenario_mock.wallclock_limit = self._time_for_task
            kwargs["stats"] = Stats(scenario_mock)
        kwargs["stats"].start_timing()

        _validate_metrics(kwargs["metrics"], kwargs["scoring_functions"])

        # Fit a pipeline, which will be stored on disk
        # which we can later load via the backend
        ta = ExecuteTaFuncWithQueue(
            backend=self._backend,
            autosklearn_seed=self._seed,
            abort_on_first_run_crash=False,
            multi_objectives=["cost"],
            cost_for_crash=get_cost_of_crash(kwargs["metrics"]),
            port=self._logger_port,
            **kwargs,
            **self._resampling_strategy_arguments,
        )

        run_info, run_value = ta.run_wrapper(
            RunInfo(
                config=config,
                instance=None,
                instance_specific=None,
                seed=self._seed,
                cutoff=kwargs.pop("cutoff", self._per_run_time_limit),
                capped=False,
            )
        )

        pipeline = None
        if kwargs["disable_file_output"] or kwargs["resampling_strategy"] == "test":
            self._logger.warning("File output is disabled. No pipeline can returned")
        elif run_value.status == StatusType.SUCCESS:
            if kwargs["resampling_strategy"] in ("cv", "cv-iterative-fit"):
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
        check_is_fitted(self)

        if (
            self._resampling_strategy
            not in ("holdout", "holdout-iterative-fit", "cv", "cv-iterative-fit")
            and not self._can_predict
        ):
            raise NotImplementedError(
                "Predict is currently not implemented for resampling "
                f"strategy {self._resampling_strategy}, please call refit()."
            )
        elif self._disable_evaluator_output is not False:
            raise NotImplementedError(
                "Predict cannot be called when evaluator output is disabled."
            )

        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        # If self.ensemble_ is None, it means that ensemble_class is None.
        # In such cases, raise error because predict and predict_proba cannot
        # be called.
        if self.ensemble_ is None:
            raise ValueError(
                "Predict and predict_proba can only be called "
                "if ensemble class is given."
            )

        # Make sure that input is valid
        if self.InputValidator is None or not self.InputValidator._is_fitted:
            raise ValueError(
                "predict() can only be called after performing fit(). Kindly call "
                "the estimator fit() method first."
            )
        X = self.InputValidator.feature_validator.transform(X)

        # Parallelize predictions across models with n_jobs processes.
        # Each process computes predictions in chunks of batch_size rows.
        try:
            for i, tmp_model in enumerate(self.models_.values()):
                # TODO, modify this
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
                raise ValueError("Found no fitted models!")

        all_predictions = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_model_predict)(
                model=models[identifier], X=X, task=self._task, batch_size=batch_size
            )
            for identifier in self.ensemble_.get_selected_model_identifiers()
        )

        if len(all_predictions) == 0:
            raise ValueError(
                "Something went wrong generating the predictions. "
                "The ensemble should consist of the following "
                "models: %s, the following models were loaded: "
                "%s"
                % (
                    str(list(self.ensemble_indices_.keys())),
                    str(list(self.models_.keys())),
                )
            )

        predictions = self.ensemble_.predict(all_predictions)

        if self._task not in REGRESSION_TASKS:
            # Make sure average prediction probabilities
            # are within a valid range
            # Individual models are checked in _model_predict
            predictions = np.clip(predictions, 0.0, 1.0)

        return predictions

    def fit_ensemble(
        self,
        y: SUPPORTED_TARGET_TYPES,
        task: Optional[int] = None,
        precision: Literal[16, 32, 64] = 32,
        dataset_name: Optional[str] = None,
        ensemble_nbest: Optional[int] = None,
        ensemble_class: Optional[AbstractEnsemble] = EnsembleSelection,
        ensemble_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Scorer | Sequence[Scorer] | None = None,
    ):
        check_is_fitted(self)

        if ensemble_class is None and self._ensemble_class is None:
            raise ValueError(
                "Please pass `ensemble_class` either to `fit_ensemble()` "
                "or the constructor."
            )

        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)

        if self._resampling_strategy in ["partial-cv", "partial-cv-iterative-fit"]:
            raise ValueError(
                "Cannot call fit_ensemble with resampling "
                "strategy %s." % self._resampling_strategy
            )

        if self._logger is None:
            self._logger = self._get_logger(dataset_name)

        # Make sure that input is valid
        y = self.InputValidator.target_validator.transform(y)

        metrics = metrics if metrics is not None else self._metrics
        if not isinstance(metrics, Sequence):
            metrics = [metrics]

        # Use the current thread to start the ensemble builder process
        # The function ensemble_builder_process will internally create a ensemble
        # builder in the provide dask client
        with self._dask as dask_client:
            manager = EnsembleBuilderManager(
                start_time=time.time(),
                time_left_for_ensembles=self._time_for_task,
                backend=copy.deepcopy(self._backend),
                dataset_name=dataset_name if dataset_name else self._dataset_name,
                task=task if task else self._task,
                metrics=metrics if metrics is not None else self._metrics,
                ensemble_class=(
                    ensemble_class
                    if ensemble_class is not None
                    else self._ensemble_class
                ),
                ensemble_kwargs=(
                    ensemble_kwargs
                    if ensemble_kwargs is not None
                    else self._ensemble_kwargs
                ),
                ensemble_nbest=ensemble_nbest
                if ensemble_nbest
                else self._ensemble_nbest,
                max_models_on_disc=self._max_models_on_disc,
                seed=self._seed,
                precision=precision if precision else self.precision,
                max_iterations=1,
                read_at_most=None,
                memory_limit=self._memory_limit,
                random_state=self._seed,
                logger_port=self._logger_port,
                pynisher_context=self._multiprocessing_context,
            )
            manager.build_ensemble(dask_client)
            future = manager.futures.pop()
            result = future.result()

        if result is None:
            raise ValueError(
                "Error building the ensemble - please check the log file and command "
                "line output for error messages."
            )
        self.ensemble_performance_history, _ = result
        self._ensemble_class = ensemble_class

        self._load_models()
        return self

    def _load_models(self):
        if self._ensemble_class is not None:
            self.ensemble_ = self._backend.load_ensemble(self._seed)
        else:
            self.ensemble_ = None

        # If no ensemble is loaded, try to get the best performing model.
        # This is triggered if
        # 1. self._ensemble_class is None (see if-statement above)
        # 2. if the ensemble builder crashed and no ensemble is available
        # 3. if the ensemble cannot be built because of arguments passed
        #    by the user (disable_evaluator_output and
        #    resampling_strategy)
        if (
            not self.ensemble_
            and not (
                self._disable_evaluator_output is True
                or (
                    isinstance(self._disable_evaluator_output, list)
                    and "model" in self._disable_evaluator_output
                )
            )
            and self._resampling_strategy
            not in (
                "partial-cv",
                "partial-cv-iterative-fit",
            )
        ):
            self.ensemble_ = self._load_best_individual_model()

        if self.ensemble_:
            identifiers = self.ensemble_.get_selected_model_identifiers()
            self.models_ = self._backend.load_models_by_identifiers(identifiers)

            if self._resampling_strategy in ("cv", "cv-iterative-fit"):
                self.cv_models_ = self._backend.load_cv_models_by_identifiers(
                    identifiers
                )
            else:
                self.cv_models_ = None
        else:
            self.models_ = []
            self.cv_models_ = []

    def _load_best_individual_model(self):
        """
        In case of failure during ensemble building,
        this method returns the single best model found
        by AutoML.
        This is a robust mechanism to be able to predict,
        even though no ensemble was found by ensemble builder.
        It is also used to load the single best model in case
        the user does not want to build an ensemble.
        """
        # We also require that the model is fit and a task is defined
        if not self._task:
            return None

        # SingleBest contains the best model found by AutoML
        ensemble = SingleBestFromRunhistory(
            metrics=self._metrics,
            task_type=self._task,
            seed=self._seed,
            run_history=self.runhistory_,
            backend=self._backend,
            random_state=self._seed,
        )
        self._logger.warning(
            "No valid ensemble was created. Please check the log"
            "file for errors. Default to the best individual estimator:{}".format(
                ensemble.get_identifiers_with_weights()[0][0]
            )
        )
        return ensemble

    def _load_pareto_set(self) -> Sequence[VotingClassifier | VotingRegressor]:
        if self.ensemble_ is None:
            self.ensemble_ = self._backend.load_ensemble(self._seed)

        # If no ensemble is loaded we cannot do anything
        if not self.ensemble_:
            raise ValueError("Pareto set only available if ensemble can be loaded.")

        if isinstance(self.ensemble_, AbstractMultiObjectiveEnsemble):
            pareto_set = self.ensemble_.pareto_set
        else:
            self._logger.warning(
                "Pareto set not available for single objective ensemble "
                "method. The Pareto set will only include the single ensemble "
                f"constructed by {type(self.ensemble_)},"
            )
            pareto_set = [self.ensemble_]

        ensembles = []
        for ensemble in pareto_set:
            identifiers = ensemble.get_selected_model_identifiers()
            weights = {
                identifier: weight
                for identifier, weight in ensemble.get_identifiers_with_weights()
            }

            if self._task in CLASSIFICATION_TASKS:
                voter = VotingClassifier(
                    estimators=None,
                    voting="soft",
                )
                kind = "classifier"
            else:
                voter = VotingRegressor(estimators=None)
                kind = "regeressor"

            if self._resampling_strategy in ("cv", "cv-iterative-fit"):
                models = self._backend.load_cv_models_by_identifiers(identifiers)
            else:
                models = self._backend.load_models_by_identifiers(identifiers)

            if len(models) == 0:
                raise ValueError("No models fitted!")

            weight_vector = []
            estimators = []
            for identifier in identifiers:
                estimator = models[identifier]
                weight = weights[identifier]

                # Kind of hacky, really the dummy models should
                # act like everything else does. Doing this is
                # required so that the VotingClassifier/Regressor
                # can use it as intended
                if not isinstance(estimator, Pipeline):
                    if kind == "classifier":
                        steps = [
                            ("data_preprocessor", None),
                            ("balancing", None),
                            ("feature_preprocessor", None),
                            (kind, estimator),
                        ]
                    else:
                        steps = [
                            ("data_preprocessor", None),
                            ("feature_preprocessor", None),
                            (kind, estimator),
                        ]

                    estimator = Pipeline(steps=steps)

                weight_vector.append(weight)
                estimators.append(estimator)

            voter.estimators = estimators
            voter.estimators_ = estimators
            voter.weights = weight_vector

            if self._task in CLASSIFICATION_TASKS:
                # Scikit-learn would raise a shape error here which we
                # have to work around.

                def inverse_transform(self, y):
                    if len(y.shape) == 1:
                        y = y.reshape((-1, 1))
                        reshaped = True
                    else:
                        reshaped = False
                    y = self.old_inverse_transform(y)
                    if reshaped:
                        return y.flatten()
                    else:
                        return y

                voter.le_ = copy.deepcopy(self.InputValidator.target_validator.encoder)
                functype = types.MethodType
                voter.le_.old_inverse_transform = voter.le_.inverse_transform
                voter.le_.inverse_transform = functype(inverse_transform, voter.le_)

            ensembles.append(voter)

        return ensembles

    def score(self, X, y):
        # fix: Consider only index 1 of second dimension
        # Don't know if the reshaping should be done there or in calculate_score

        # Predict has validate within it, so we
        # call it before the upcoming validate call
        # The reason is we do not want to trigger the
        # check for changing input types on successive
        # input validator calls
        check_is_fitted(self)
        prediction = self.predict(X)
        y = self.InputValidator.target_validator.transform(y)

        # Encode the prediction using the input validator
        # We train autosklearn with a encoded version of y,
        # which is decoded by predict().
        # Above call to validate() encodes the y given for score()
        # Below call encodes the prediction, so we compare in the
        # same representation domain
        prediction = self.InputValidator.target_validator.transform(prediction)

        return compute_single_metric(
            solution=y,
            prediction=prediction,
            task_type=self._task,
            metric=self._metrics[0],
        )

    def _get_runhistory_models_performance(self):
        metric = self._metrics[0]
        data = self.runhistory_.data
        performance_list = []
        for run_key, run_value in data.items():
            if run_value.status != StatusType.SUCCESS:
                # Ignore crashed runs
                continue
            # Alternatively, it is possible to also obtain the start time with
            # ``run_value.starttime``
            endtime = pd.Timestamp(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_value.endtime))
            )
            cost = run_value.cost
            train_loss = run_value.additional_info["train_loss"]
            if len(self._metrics) > 1:
                cost = cost[0]
                train_loss = train_loss[0]
            val_score = metric._optimum - (metric._sign * cost)
            train_score = metric._optimum - (metric._sign * train_loss)
            scores = {
                "Timestamp": endtime,
                "single_best_optimization_score": val_score,
                "single_best_train_score": train_score,
            }
            # Append test-scores, if data for test_loss are available.
            # This is the case, if X_test and y_test where provided.
            if "test_loss" in run_value.additional_info:
                test_loss = run_value.additional_info["test_loss"]
                if len(self._metrics) > 1:
                    test_loss = test_loss[0]
                test_score = metric._optimum - (metric._sign * test_loss)
                scores["single_best_test_score"] = test_score

            performance_list.append(scores)
        return pd.DataFrame(performance_list)

    @property
    def performance_over_time_(self):
        check_is_fitted(self)
        individual_performance_frame = self._get_runhistory_models_performance()
        best_values = pd.Series(
            {
                "single_best_optimization_score": -np.inf,
                "single_best_test_score": -np.inf,
                "single_best_train_score": -np.inf,
            }
        )
        for idx in individual_performance_frame.index:
            if (
                individual_performance_frame.loc[idx, "single_best_optimization_score"]
                > best_values["single_best_optimization_score"]
            ):
                best_values = individual_performance_frame.loc[idx]
            individual_performance_frame.loc[idx] = best_values

        performance_over_time = individual_performance_frame

        if self._ensemble_class is not None:
            ensemble_performance_frame = pd.DataFrame(self.ensemble_performance_history)
            best_values = pd.Series(
                {"ensemble_optimization_score": -np.inf, "ensemble_test_score": -np.inf}
            )
            for idx in ensemble_performance_frame.index:
                if (
                    ensemble_performance_frame.loc[idx, "ensemble_optimization_score"]
                    > best_values["ensemble_optimization_score"]
                ):
                    best_values = ensemble_performance_frame.loc[idx]
                ensemble_performance_frame.loc[idx] = best_values

            performance_over_time = (
                pd.merge(
                    ensemble_performance_frame,
                    individual_performance_frame,
                    on="Timestamp",
                    how="outer",
                )
                .sort_values("Timestamp")
                .fillna(method="ffill")
            )

        return performance_over_time

    @property
    def cv_results_(self):
        check_is_fitted(self)
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
        if self._resampling_strategy in ["partial-cv", "partial-cv-iterative-fit"]:
            raise ValueError("Cannot call cv_results when using partial-cv!")

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

        for metric in itertools.chain(self._metrics, self._scoring_functions):
            metric_dict[metric.name] = []
            metric_mask[metric.name] = []

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
                status.append("Success")
            elif s == StatusType.DONOTADVANCE:
                status.append("Success (but do not advance to higher budget)")
            elif s == StatusType.TIMEOUT:
                status.append("Timeout")
            elif s == StatusType.CRASHED:
                status.append("Crash")
            elif s == StatusType.ABORT:
                status.append("Abort")
            elif s == StatusType.MEMOUT:
                status.append("Memout")
            # TODO remove StatusType.RUNNING at some point in the future when the new
            # SMAC 0.13.2 is the new minimum required version!
            elif s in (StatusType.STOP, StatusType.RUNNING):
                continue
            else:
                raise NotImplementedError(s)

            param_dict = config.get_dictionary()
            params.append(param_dict)

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

            cost = [run_value.cost] if len(self._metrics) == 1 else run_value.cost
            for metric_idx, metric in enumerate(self._metrics):
                metric_cost = cost[metric_idx]
                metric_value = metric._optimum - (metric._sign * metric_cost)
                mask_value = False
                metric_dict[metric.name].append(metric_value)
                metric_mask[metric.name].append(mask_value)

            optimization_metric_names = set(m.name for m in self._metrics)
            for metric in self._scoring_functions:
                if metric.name in optimization_metric_names:
                    continue
                if metric.name in run_value.additional_info.keys():
                    metric_cost = run_value.additional_info[metric.name]
                    metric_value = metric._optimum - (metric._sign * metric_cost)
                    mask_value = False
                else:
                    metric_value = np.NaN
                    mask_value = True
                metric_dict[metric.name].append(metric_value)
                metric_mask[metric.name].append(mask_value)

        if len(self._metrics) == 1:
            results["mean_test_score"] = np.array(metric_dict[self._metrics[0].name])
            rank_order = -1 * self._metrics[0]._sign * results["mean_test_score"]
            results["rank_test_scores"] = scipy.stats.rankdata(rank_order, method="min")
        else:
            for metric in self._metrics:
                key = f"mean_test_{metric.name}"
                results[key] = np.array(metric_dict[metric.name])
                rank_order = -1 * metric._sign * results[key]
                results[f"rank_test_{metric.name}"] = scipy.stats.rankdata(
                    rank_order, method="min"
                )
        for metric in self._scoring_functions:
            masked_array = ma.MaskedArray(
                metric_dict[metric.name], metric_mask[metric.name]
            )
            results[f"metric_{metric.name}"] = masked_array

        results["mean_fit_time"] = np.array(mean_fit_time)
        results["params"] = params
        results["status"] = status
        results["budgets"] = budgets

        for hp_name in hp_names:
            masked_array = ma.MaskedArray(
                parameter_dictionaries[hp_name], masks[hp_name]
            )
            results["param_%s" % hp_name] = masked_array

        return results

    def sprint_statistics(self) -> str:
        check_is_fitted(self)
        cv_results = self.cv_results_
        sio = io.StringIO()
        sio.write("auto-sklearn results:\n")
        sio.write("  Dataset name: %s\n" % self._dataset_name)
        if len(self._metrics) == 1:
            sio.write("  Metric: %s\n" % self._metrics[0])
        else:
            sio.write("  Metrics: %s\n" % self._metrics)
        idx_success = np.where(
            np.array(
                [
                    status
                    in ["Success", "Success (but do not advance to higher budget)"]
                    for status in cv_results["status"]
                ]
            )
        )[0]
        if len(idx_success) > 0:
            key = (
                "mean_test_score"
                if len(self._metrics) == 1
                else f"mean_test_" f"{self._metrics[0].name}"
            )

            if not self._metrics[0]._optimum:
                idx_best_run = np.argmin(cv_results[key][idx_success])
            else:
                idx_best_run = np.argmax(cv_results[key][idx_success])
            best_score = cv_results[key][idx_success][idx_best_run]
            sio.write("  Best validation score: %f\n" % best_score)
        num_runs = len(cv_results["status"])
        sio.write("  Number of target algorithm runs: %d\n" % num_runs)
        num_success = sum(
            [
                s in ["Success", "Success (but do not advance to higher budget)"]
                for s in cv_results["status"]
            ]
        )
        sio.write("  Number of successful target algorithm runs: %d\n" % num_success)
        num_crash = sum([s == "Crash" for s in cv_results["status"]])
        sio.write("  Number of crashed target algorithm runs: %d\n" % num_crash)
        num_timeout = sum([s == "Timeout" for s in cv_results["status"]])
        sio.write(
            "  Number of target algorithms that exceeded the time "
            "limit: %d\n" % num_timeout
        )
        num_memout = sum([s == "Memout" for s in cv_results["status"]])
        sio.write(
            "  Number of target algorithms that exceeded the memory "
            "limit: %d\n" % num_memout
        )
        return sio.getvalue()

    def get_models_with_weights(self) -> list[Tuple[float, BasePipeline]]:
        check_is_fitted(self)
        if self.models_ is None or len(self.models_) == 0 or self.ensemble_ is None:
            self._load_models()

        return self.ensemble_.get_models_with_weights(self.models_)

    def show_models(self) -> dict[int, Any]:
        """Returns a dictionary containing dictionaries of ensemble models.

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
                    'data_preprocessor': <pipeline.components.data_preprocessing....>,
                    'feature_preprocessor': <autosklearn.pipeline.components....>,
                    'regressor': <autosklearn.pipeline.components.regression....>,
                    'sklearn_regressor': ARDRegression(alpha_1=0.027,...)
                    },
                ...
            }

        Returns
        -------
        dict[int, Any] : dictionary of length = number of models in the ensemble
            A dictionary of models in the ensemble, where ``model_id`` is the key.
        """  # noqa: E501
        check_is_fitted(self)

        ensemble_dict = {}

        if self._ensemble_class is None:
            warnings.warn(
                "No models in the ensemble. Kindly provide an ensemble class."
            )
            return ensemble_dict

        # check for condition when ensemble_size > 0 but there is no ensemble to load
        if self.ensemble_ is None:
            warnings.warn("No ensemble found. Returning empty dictionary.")
            return ensemble_dict

        def has_key(rv, key):
            return rv.additional_info and key in rv.additional_info

        table_dict = {}
        for run_key, run_val in self.runhistory_.data.items():
            if has_key(run_val, "num_run"):
                model_id = run_val.additional_info["num_run"]
                table_dict[model_id] = {"model_id": model_id, "cost": run_val.cost}

        # Checking if the dictionary is empty
        if not table_dict:
            raise RuntimeError(
                "No model found. Try increasing 'time_left_for_this_task'."
            )

        for (_, model_id, _), weight in self.ensemble_.get_identifiers_with_weights():
            table_dict[model_id]["ensemble_weight"] = weight

        table = pd.DataFrame.from_dict(table_dict, orient="index")
        table.sort_values(by="cost", inplace=True)

        # Check which resampling strategy is chosen and selecting the appropriate models
        is_cv = self._resampling_strategy == "cv"
        models = self.cv_models_ if is_cv else self.models_

        rank = 1  # Initializing rank for the first model
        for (_, model_id, _), model in models.items():
            model_dict = {}  # Declaring model dictionary

            # Inserting model_id, rank, cost and ensemble weight
            model_dict["model_id"] = table.loc[model_id]["model_id"].astype(int)
            model_dict["rank"] = rank
            model_dict["cost"] = table.loc[model_id]["cost"]
            model_dict["ensemble_weight"] = table.loc[model_id]["ensemble_weight"]
            rank += 1  # Incrementing rank by 1 for the next model

            # The steps in the models pipeline are as follows:
            # 'data_preprocessor': DataPreprocessor,
            # 'balancing': Balancing,
            # 'feature_preprocessor': FeaturePreprocessorChoice,
            # 'classifier'/'regressor': ClassifierChoice/RegressorChoice (wrapped model)

            # For 'cv' (cross validation) strategy
            if is_cv:
                # Voting model created by cross validation
                cv_voting_ensemble = model
                model_dict["voting_model"] = cv_voting_ensemble

                # List of models, each trained on one cv fold
                cv_models = []
                for cv_model in cv_voting_ensemble.estimators_:
                    estimator = dict(cv_model.steps)

                    # Adding sklearn model to the model dictionary
                    model_type, autosklearn_wrapped_model = cv_model.steps[-1]
                    estimator[
                        f"sklearn_{model_type}"
                    ] = autosklearn_wrapped_model.choice.estimator
                    cv_models.append(estimator)
                model_dict["estimators"] = cv_models

            # For any other strategy
            else:
                steps = dict(model.steps)
                model_dict.update(steps)

                # Adding sklearn model to the model dictionary
                model_type, autosklearn_wrapped_model = model.steps[-1]
                model_dict[
                    f"sklearn_{model_type}"
                ] = autosklearn_wrapped_model.choice.estimator

            # Insterting model_dict in the ensemble dictionary
            ensemble_dict[model_id] = model_dict

        return ensemble_dict

    def has_ensemble(self) -> bool:
        """
        Returns
        -------
        bool
            Whether this AutoML instance has an ensemble
        """
        return self.ensemble_ is not None

    def _create_search_space(
        self,
        tmp_dir: str,
        backend: Backend,
        datamanager: XYDataManager,
        include: Optional[Mapping[str, list[str]]] = None,
        exclude: Optional[Mapping[str, list[str]]] = None,
    ) -> Tuple[ConfigurationSpace, str]:
        configspace_path = os.path.join(tmp_dir, "space.json")
        configuration_space = pipeline.get_configuration_space(
            datamanager,
            include=include,
            exclude=exclude,
        )
        backend.write_txt_file(
            configspace_path,
            cs_json.write(configuration_space),
            "Configuration space",
        )

        return configuration_space, configspace_path

    def __getstate__(self) -> dict[str, Any]:
        # Cannot serialize a client!
        self._dask = None
        self.logging_server = None
        self.stop_logging_server = None
        return self.__dict__

    def __del__(self) -> None:
        # Clean up the logger
        self._clean_logger()


class AutoMLClassifier(AutoML):

    _task_mapping = {
        "multilabel-indicator": MULTILABEL_CLASSIFICATION,
        "multiclass": MULTICLASS_CLASSIFICATION,
        "binary": BINARY_CLASSIFICATION,
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
        y: SUPPORTED_TARGET_TYPES | spmatrix,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES | spmatrix] = None,
        feat_type: Optional[list[str]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ) -> AutoMLClassifier:
        return super().fit(
            X,
            y,
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
        y: SUPPORTED_TARGET_TYPES | spmatrix,
        config: Configuration | dict[str, str | float | int],
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES | spmatrix] = None,
        feat_type: Optional[list[str]] = None,
        **kwargs,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        return super().fit_pipeline(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            dataset_name=dataset_name,
            config=config,
            is_classification=True,
            feat_type=feat_type,
            **kwargs,
        )

    def predict(
        self,
        X: SUPPORTED_FEAT_TYPES,
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
    ) -> np.ndarray:
        check_is_fitted(self)

        predicted_probabilities = super().predict(
            X, batch_size=batch_size, n_jobs=n_jobs
        )

        if self.InputValidator.target_validator.is_single_column_target():
            predicted_indexes = np.argmax(predicted_probabilities, axis=1)
        else:
            predicted_indexes = (predicted_probabilities > 0.5).astype(int)

        return self.InputValidator.target_validator.inverse_transform(predicted_indexes)

    def predict_proba(
        self,
        X: SUPPORTED_FEAT_TYPES,
        batch_size: Optional[int] = None,
        n_jobs: int = 1,
    ) -> np.ndarray:
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)


class AutoMLRegressor(AutoML):

    _task_mapping = {
        "continuous-multioutput": MULTIOUTPUT_REGRESSION,
        "continuous": REGRESSION,
        "multiclass": REGRESSION,
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
        y: SUPPORTED_TARGET_TYPES | spmatrix,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES | spmatrix] = None,
        feat_type: Optional[list[str]] = None,
        dataset_name: Optional[str] = None,
        only_return_configuration_space: bool = False,
        load_models: bool = True,
    ) -> AutoMLRegressor:
        return super().fit(
            X,
            y,
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
        y: SUPPORTED_TARGET_TYPES | spmatrix,
        config: Configuration | dict[str, str | float | int],
        dataset_name: Optional[str] = None,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES | spmatrix] = None,
        feat_type: Optional[list[str]] = None,
        **kwargs: dict,
    ) -> Tuple[Optional[BasePipeline], RunInfo, RunValue]:
        return super().fit_pipeline(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            config=config,
            feat_type=feat_type,
            dataset_name=dataset_name,
            is_classification=False,
            **kwargs,
        )
