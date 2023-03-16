from __future__ import annotations

from abc import ABC
from typing import Sequence, Tuple, Union

import copy
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from logging import Logger
from pathlib import Path

import numpy as np
import pynisher
from ConfigSpace import ConfigurationSpace
from dask.distributed import Client
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit
from sklearn.model_selection._split import _RepeatedSplits
from smac import AlgorithmConfigurationFacade, Callback, RunHistory, Scenario
from smac.facade import AbstractFacade
from smac.initial_design import DefaultInitialDesign
from smac.intensifier import Intensifier
from smac.multi_objective import ParEGO
from smac.runhistory.dataclasses import TrajectoryItem
from smac.runhistory.encoder import RunHistoryLogEncoder
from smac.runner import AbstractRunner, DaskParallelRunner
from smac.runner.abstract_serial_runner import AbstractSerialRunner

import autosklearn
from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    CLASSIFICATION_TASKS,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    TASK_TYPES_TO_STRING,
)
from autosklearn.ensemble_building import EnsembleBuilderManager
from autosklearn.evaluation import TargetFunctionRunnerWithQueue
from autosklearn.metalearning.metafeatures.metafeatures import (
    calculate_all_metafeatures_encoded_labels,
    calculate_all_metafeatures_with_labels,
)
from autosklearn.metalearning.metalearning.meta_base import MetaBase
from autosklearn.metalearning.mismbo import suggest_via_metalearning
from autosklearn.metrics import Scorer, get_cost_of_crash
from autosklearn.util.logging_ import PicklableClientLogger, get_named_client_logger
from autosklearn.util.parallel import preload_modules
from autosklearn.util.stopwatch import StopWatch
from autosklearn.util.str_types import (
    BUDGET_TYPE,
    PYNISHER_CONTEXT,
    RESAMPLING_STRATEGY,
)

EXCLUDE_META_FEATURES_CLASSIFICATION = {
    "Landmark1NN",
    "LandmarkDecisionNodeLearner",
    "LandmarkDecisionTree",
    "LandmarkLDA",
    "LandmarkNaiveBayes",
    "LandmarkRandomNodeLearner",
    "PCAFractionOfComponentsFor95PercentVariance",
    "PCAKurtosisFirstPC",
    "PCASkewnessFirstPC",
    "PCA",
}

EXCLUDE_META_FEATURES_REGRESSION = {
    "Landmark1NN",
    "LandmarkDecisionNodeLearner",
    "LandmarkDecisionTree",
    "LandmarkLDA",
    "LandmarkNaiveBayes",
    "PCAFractionOfComponentsFor95PercentVariance",
    "PCAKurtosisFirstPC",
    "PCASkewnessFirstPC",
    "NumberOfClasses",
    "ClassOccurences",
    "ClassProbabilityMin",
    "ClassProbabilityMax",
    "ClassProbabilityMean",
    "ClassProbabilitySTD",
    "ClassEntropy",
    "LandmarkRandomNodeLearner",
    "PCA",
}


def get_send_warnings_to_logger(logger):
    def _send_warnings_to_log(message, category, filename, lineno, file, line):
        logger.debug("%s:%s: %s:%s", filename, lineno, category.__name__, message)

    return _send_warnings_to_log


# metalearning helpers
def _calculate_metafeatures(
    data_feat_type,
    data_info_task,
    basename,
    x_train,
    y_train,
    stopwatch: StopWatch,
    logger_,
):
    with warnings.catch_warnings():
        warnings.showwarning = get_send_warnings_to_logger(logger_)

        # == Calculate metafeatures
        with stopwatch.time("Calculate meta-features") as task_timer:
            EXCLUDE_META_FEATURES = (
                EXCLUDE_META_FEATURES_CLASSIFICATION
                if data_info_task in CLASSIFICATION_TASKS
                else EXCLUDE_META_FEATURES_REGRESSION
            )

            if data_info_task in [
                MULTICLASS_CLASSIFICATION,
                BINARY_CLASSIFICATION,
                MULTILABEL_CLASSIFICATION,
                REGRESSION,
                MULTIOUTPUT_REGRESSION,
            ]:
                logger_.info("Start calculating metafeatures for %s", basename)
                result = calculate_all_metafeatures_with_labels(
                    x_train,
                    y_train,
                    feat_type=data_feat_type,
                    dataset_name=basename,
                    dont_calculate=EXCLUDE_META_FEATURES,
                    logger=logger_,
                )
                for key in list(result.metafeature_values.keys()):
                    if result.metafeature_values[key].type_ != "METAFEATURE":
                        del result.metafeature_values[key]

            else:
                result = None
                logger_.info("Metafeatures not calculated")

        logger_.info(f"{task_timer.name} took {task_timer.wall_duration:5.2f}"),
        return result


def _calculate_metafeatures_encoded(
    data_feat_type,
    basename,
    x_train,
    y_train,
    stopwatch: StopWatch,
    task,
    logger_,
):
    with warnings.catch_warnings():
        warnings.showwarning = get_send_warnings_to_logger(logger_)

        EXCLUDE_META_FEATURES = (
            EXCLUDE_META_FEATURES_CLASSIFICATION
            if task in CLASSIFICATION_TASKS
            else EXCLUDE_META_FEATURES_REGRESSION
        )

        with stopwatch.time("Calculate meta-features encoded") as task_timer:
            result = calculate_all_metafeatures_encoded_labels(
                x_train,
                y_train,
                feat_type=data_feat_type,
                dataset_name=basename,
                dont_calculate=EXCLUDE_META_FEATURES,
                logger=logger_,
            )
            for key in list(result.metafeature_values.keys()):
                if result.metafeature_values[key].type_ != "METAFEATURE":
                    del result.metafeature_values[key]

        logger_.info(f"{task_timer.name} took {task_timer.wall_duration:5.2f}sec")
        return result


def _get_metalearning_configurations(
    meta_base,
    basename,
    metric,
    configuration_space,
    task,
    initial_configurations_via_metalearning,
    stopwatch: StopWatch,
    is_sparse,
    logger,
):
    try:
        metalearning_configurations = suggest_via_metalearning(
            meta_base,
            basename,
            metric,
            task,
            is_sparse == 1,
            initial_configurations_via_metalearning,
            logger=logger,
        )
    except Exception as e:
        logger.error("Error getting metalearning configurations!")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        metalearning_configurations = []

    return metalearning_configurations


class AutoMLOptimizer(ABC):
    """The optimizer that searches for the best performing pipeline on the data.

    It essentially just sets up SMAC and exposes the run_smbo() method to start an
    optimization run on the data stored inside the datamanager.
    """

    def __init__(
        self,
        config_space: ConfigurationSpace,
        dataset_name: str = None,
        backend: Backend = None,
        total_walltime_limit: int = None,
        func_eval_time_limit: int = None,
        memory_limit: int = None,
        metrics: Sequence[Scorer] = None,
        stopwatch: StopWatch = None,
        n_jobs: int = 1,
        dask_client: Client = None,
        port: int = None,
        initial_num_run: int = 1,
        num_metalearning_cfgs: int = 25,
        seed: int = 1,
        metadata_directory: str = None,
        resampling_strategy: Union[
            RESAMPLING_STRATEGY, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit
        ] = "holdout",
        resampling_strategy_args: dict = None,
        include: dict[str, list[str]] | None = None,
        exclude: dict[str, list[str]] | None = None,
        disable_file_output: bool = False,
        smac_scenario_args: dict[str, any] | Scenario | None = None,
        smac_facade: AbstractFacade | None = None,
        scoring_functions: list[Scorer] | None = None,
        pynisher_context: PYNISHER_CONTEXT | str | None = "spawn",
        ensemble_callback: EnsembleBuilderManager | None = None,
        trials_callback: Callback | None = None,
    ):
        self.dataset_name = dataset_name
        self.datamanager = None
        self.metrics = metrics
        self.task = None
        self.backend = backend
        self.port = port
        self.config_space = config_space
        self.n_jobs = n_jobs
        self.dask_client = dask_client
        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args
        self.worst_possible_result = get_cost_of_crash(self.metrics)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.stopwatch = stopwatch
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.initial_num_run = initial_num_run
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self._smac_facade = smac_facade
        self.scoring_functions = scoring_functions
        self.pynisher_context = pynisher_context
        self.ensemble_callback = ensemble_callback
        self.trials_callback = trials_callback

        self.logger = self._create_logger(port)

        if self.resampling_strategy is None:
            self.resampling_strategy = {}

        if self._smac_facade is None:
            self._smac_facade = AutoMLOptimizer._default_smac_facade

    def run(self) -> Tuple[RunHistory, list[TrajectoryItem], BUDGET_TYPE]:
        """Runs SMAC, the configured underlying optimizer, on the given task.

        This will return a history of configurations that were sampled by the
        optimizer and a trajectory of the incumbent (best seen so far) configuration.

        Returns
        -------
        runhistory: RunHistory
            A sequence of runs, where each element contains information about the run

        trajectory: Sequence[TrajectoryItem]
            A sequence that contains the best configuration found at each time step
            in the optimization

        budget_type: BUDGET_TYPE
            The budget type, based on which the corresponding target algorithm is chosen
        """
        self.stopwatch.start("SMBO")

        self._load_data_manager()
        self.config_space.seed(self.seed)

        startup_time = self.stopwatch.time_since(self.dataset_name, "start")
        total_walltime_limit = self.total_walltime_limit - startup_time - 5

        # Initialize the optimizer
        smac = self._initialize_smac(total_walltime_limit)

        # Run the optimizer
        smac.optimize()

        # Return the information about the optimization run
        runhistory = smac.runhistory
        trajectory = smac.intensifier.trajectory

        # Get the type of budget used by the runner
        # (We use a custom object as our target function runner, which has a budget_type
        # attribute)
        if isinstance(smac._runner, DaskParallelRunner):
            _budget_type = smac._runner._single_worker.budget_type
        elif isinstance(smac._runner, AbstractSerialRunner):
            _budget_type = smac._runner.budget_type
        else:
            raise NotImplementedError(type(smac._runner))

        self.stopwatch.stop("SMBO")

        return runhistory, trajectory, _budget_type

    def _initialize_smac(self, total_walltime_limit: float) -> AbstractFacade:
        """Initialize the optimizer with everything needed to start an optimization run.

        Parameters
        ----------
        total_walltime_limit: float
            The time limit to run the optimization under

        Returns
        -------
        smac: AbstractFacade
            The configured optimizer
        """
        # Ask for a set of initial configurations for the optimizer
        initial_configurations = self.get_metalearning_suggestions()

        # Set up the environment for the optimizer
        scenario = self._create_scenario(total_walltime_limit)

        # Create the target function runner that executes the target function
        target_function_runner = TargetFunctionRunnerWithQueue(
            scenario=scenario,
            backend=copy.deepcopy(self.backend),
            autosklearn_seed=self.seed,
            resampling_strategy=self.resampling_strategy,
            initial_num_run=self.initial_num_run,
            include=self.include,
            exclude=self.exclude,
            metrics=self.metrics,
            memory_limit=self.memory_limit,
            disable_file_output=self.disable_file_output,
            scoring_functions=self.scoring_functions,
            port=self.port,
            pynisher_context=self.pynisher_context,
            cost_for_crash=self.worst_possible_result,
            **self.resampling_strategy_args,
        )

        # Wrap target function iside a dask runner that uses the client.
        # This used to be a functionality provided by SMAC when a dask client was
        # passed in as a parameter for a SMAC facade. However, it's not offered by SMAC
        # anymore, so we replicate the old funcitonality here, because we rely on it.
        target_function_runner = DaskParallelRunner(
            single_worker=target_function_runner,
            dask_client=self.dask_client,
        )

        # Configure the optimizer, SMAC
        smac_facade_args = {
            "scenario": scenario,
            "target_function": target_function_runner,
            "metalearning_configurations": initial_configurations,
            "multi_objective_algorithm": None,
            "callbacks": [],
            "seed": self.seed,
        }

        if len(self.metrics) > 1:
            # rho should be set to 0.05
            smac_facade_args["multi_objective_algorithm"] = ParEGO

        if self.ensemble_callback is not None:
            smac_facade_args["callbacks"].append(self.ensemble_callback)
        if self.trials_callback is not None:
            smac_facade_args["callbacks"].append(self.trials_callback)

        if self._smac_facade is not None:
            smac = self._smac_facade(**smac_facade_args)
        else:
            smac = self._default_smac_facade(**smac_facade_args)

        return smac

    def _create_logger(self, port: int | None) -> Logger | PicklableClientLogger:
        """Creates the logger.

        Parameters
        ----------
        port: int
            The port of the logging server

        Returns
        -------
        logger
        """
        if port is None:
            return logging.getLogger(__name__)
        logger_name = f"{type(self).__name__}({self.seed}):{'' or self.dataset_name}"
        return get_named_client_logger(name=logger_name, port=self.port)

    def _load_data_manager(self) -> None:
        """Loads the object that stores the dataset and some other information about
        the data."""
        self.datamanager = self.backend.load_datamanager()
        self.task = self.datamanager.info["task"]

    def _create_scenario(self, total_walltime_limit: float) -> Scenario:
        """Sets up the optimization environment for the optimizer.

        Parameters
        ----------
        total_walltime_limit: int
            the total real time to run the optimization for

        Returns
        -------
        scenario: Scenario
            the object describing the optimization environment
        """
        objectives = "cost"
        if len(self.metrics) > 1:
            objectives = [metric.name for metric in self.metrics]

        # An instance represents a specific scenario/condition (e.g. different
        # datasets, subsets, transformations) for the target function to run on.
        if self.resampling_strategy in ["partial-cv", "partial-cv-iterative-fit"]:
            num_folds = self.resampling_strategy_args["folds"]
            instances = [
                json.dumps({"task_id": self.dataset_name, "fold": fold_number})
                for fold_number in range(num_folds)
            ]
        else:
            instances = [json.dumps({"task_id": self.dataset_name})]

        scenario = Scenario(
            configspace=self.config_space,
            name=self.dataset_name,
            output_directory=Path(self.backend.get_smac_output_directory()),
            deterministic=True,
            objectives=objectives,
            crash_cost=self.worst_possible_result,
            termination_cost_threshold=np.inf,
            walltime_limit=total_walltime_limit,
            trial_walltime_limit=self.func_eval_time_limit,
            trial_memory_limit=self.memory_limit,
            n_trials=sys.maxsize,
            instances=instances,
            instance_features=None,
            min_budget=None,
            max_budget=None,
            seed=self.seed,
            n_workers=self.n_jobs,
        )
        return scenario

    def get_metalearning_suggestions(self):
        # == METALEARNING suggestions
        # we start by evaluating the defaults on the full dataset again
        # and add the suggestions from metalearning behind it
        if self.num_metalearning_cfgs > 0:
            # If metadata directory is None, use default
            if self.metadata_directory is None:
                metalearning_directory = os.path.dirname(
                    autosklearn.metalearning.__file__
                )
                # There is no multilabel data in OpenML
                if self.task == MULTILABEL_CLASSIFICATION:
                    meta_task = BINARY_CLASSIFICATION
                else:
                    meta_task = self.task
                metadata_directory = os.path.join(
                    metalearning_directory,
                    "files",
                    "%s_%s_%s"
                    % (
                        self.metrics[0],
                        TASK_TYPES_TO_STRING[meta_task],
                        "sparse" if self.datamanager.info["is_sparse"] else "dense",
                    ),
                )
                self.metadata_directory = metadata_directory

            # If metadata directory is specified by user,
            # then verify that it exists.
            else:
                if not os.path.exists(self.metadata_directory):
                    raise ValueError(
                        "The specified metadata directory '%s' "
                        "does not exist!" % self.metadata_directory
                    )

                else:
                    # There is no multilabel data in OpenML
                    if self.task == MULTILABEL_CLASSIFICATION:
                        meta_task = BINARY_CLASSIFICATION
                    else:
                        meta_task = self.task

                    metadata_directory = os.path.join(
                        self.metadata_directory,
                        "%s_%s_%s"
                        % (
                            self.metrics[0],
                            TASK_TYPES_TO_STRING[meta_task],
                            "sparse" if self.datamanager.info["is_sparse"] else "dense",
                        ),
                    )
                    # Check that the metadata directory has the correct
                    # subdirectory needed for this dataset.
                    if os.path.basename(metadata_directory) not in os.listdir(
                        self.metadata_directory
                    ):
                        raise ValueError(
                            "The specified metadata directory "
                            "'%s' does not have the correct "
                            "subdirectory '%s'"
                            % (
                                self.metadata_directory,
                                os.path.basename(metadata_directory),
                            )
                        )
                self.metadata_directory = metadata_directory

            if os.path.exists(self.metadata_directory):
                self.logger.info("Metadata directory: %s", self.metadata_directory)
                meta_base = MetaBase(
                    self.config_space, self.metadata_directory, self.logger
                )

                metafeature_calculation_time_limit = int(self.total_walltime_limit / 4)
                metafeature_calculation_start_time = time.time()
                meta_features = self._calculate_metafeatures_with_limits(
                    metafeature_calculation_time_limit
                )
                metafeature_calculation_end_time = time.time()
                metafeature_calculation_time_limit = (
                    metafeature_calculation_time_limit
                    - (
                        metafeature_calculation_end_time
                        - metafeature_calculation_start_time
                    )
                )

                if metafeature_calculation_time_limit < 1:
                    self.logger.warning(
                        "Time limit for metafeature calculation less "
                        "than 1 seconds (%f). Skipping calculation "
                        "of metafeatures for encoded dataset.",
                        metafeature_calculation_time_limit,
                    )
                    meta_features_encoded = None
                else:
                    with warnings.catch_warnings():
                        warnings.showwarning = get_send_warnings_to_logger(self.logger)
                    meta_features_encoded = (
                        self._calculate_metafeatures_encoded_with_limits(
                            metafeature_calculation_time_limit
                        )
                    )

                # In case there is a problem calculating the encoded meta-features
                if meta_features is None:
                    if meta_features_encoded is not None:
                        meta_features = meta_features_encoded
                else:
                    if meta_features_encoded is not None:
                        meta_features.metafeature_values.update(
                            meta_features_encoded.metafeature_values
                        )

                if meta_features is not None:
                    meta_base.add_dataset(self.dataset_name, meta_features)
                    # Do mean imputation of the meta-features - should be done specific
                    # for each prediction model!
                    all_metafeatures = meta_base.get_metafeatures(
                        features=list(meta_features.keys())
                    )
                    all_metafeatures.fillna(all_metafeatures.mean(), inplace=True)

                    with warnings.catch_warnings():
                        warnings.showwarning = get_send_warnings_to_logger(self.logger)
                        metalearning_configurations = (
                            self.collect_metalearning_suggestions(meta_base)
                        )
                    if metalearning_configurations is None:
                        metalearning_configurations = []
                    self._load_data_manager()

                    self.logger.info("%s", meta_features)

                    # Convert meta-features into a dictionary because the scenario
                    # expects a dictionary
                    meta_features_dict = {}
                    for dataset, series in all_metafeatures.iterrows():
                        meta_features_dict[dataset] = series.values
                    meta_features_list = []
                    for meta_feature_name in all_metafeatures.columns:
                        meta_features_list.append(
                            meta_features[meta_feature_name].value
                        )
                    self.logger.info(list(meta_features_dict.keys()))

            else:
                meta_features = None
                self.logger.warning(
                    "Could not find meta-data directory %s" % metadata_directory
                )

        else:
            meta_features = None
        if meta_features is None:
            metalearning_configurations = []
        return metalearning_configurations

    def collect_metalearning_suggestions(self, meta_base):
        with self.stopwatch.time("Initial Configurations") as task:
            metalearning_configurations = _get_metalearning_configurations(
                meta_base=meta_base,
                basename=self.dataset_name,
                metric=self.metrics[0],
                configuration_space=self.config_space,
                task=self.task,
                is_sparse=self.datamanager.info["is_sparse"],
                initial_configurations_via_metalearning=self.num_metalearning_cfgs,
                stopwatch=self.stopwatch,
                logger=self.logger,
            )

        self.logger.debug(f"Initial Configurations: {len(metalearning_configurations)}")
        for config in metalearning_configurations:
            self.logger.debug(config)

        self.logger.debug(f"{task.name} took {task.wall_duration:5.2f}sec")

        time_since_start = self.stopwatch.time_since(self.dataset_name, "start")
        time_left = self.total_walltime_limit - time_since_start
        self.logger.info(f"Time left for {task.name}: {time_left:5.2f}s")

        return metalearning_configurations

    def _calculate_metafeatures_with_limits(self, time_limit):
        res = None
        time_limit = max(time_limit, 1)
        try:
            context = multiprocessing.get_context(self.pynisher_context)
            preload_modules(context)

            memory_limit = None
            if self.memory_limit is not None:
                memory_limit = (self.memory_limit, "MB")

            safe_mf = pynisher.limit(
                _calculate_metafeatures,
                memory=memory_limit,
                wall_time=int(time_limit),
                # grace_period_in_s=30,
                context=context,
                # logger=self.logger,
            )
            res = safe_mf(
                data_feat_type=self.datamanager.feat_type,
                data_info_task=self.datamanager.info["task"],
                x_train=self.datamanager.data["X_train"],
                y_train=self.datamanager.data["Y_train"],
                basename=self.dataset_name,
                stopwatch=self.stopwatch,
                logger_=self.logger,
            )
        except Exception as e:
            self.logger.error("Error getting metafeatures: %s", str(e))

        return res

    def _calculate_metafeatures_encoded_with_limits(self, time_limit):
        res = None
        time_limit = max(time_limit, 1)
        try:
            context = multiprocessing.get_context(self.pynisher_context)
            preload_modules(context)

            memory_limit = None
            if self.memory_limit is not None:
                memory_limit = (self.memory_limit, "MB")

            # updatesmac: figure out what the parameters should be
            safe_mf = pynisher.limit(
                _calculate_metafeatures_encoded,
                memory=memory_limit,
                wall_time=int(time_limit),
                # grace_period_in_s=30,
                context=context,
                # logger=self.logger,
            )
            res = safe_mf(
                data_feat_type=self.datamanager.feat_type,
                task=self.datamanager.info["task"],
                x_train=self.datamanager.data["X_train"],
                y_train=self.datamanager.data["Y_train"],
                basename=self.dataset_name,
                stopwatch=self.stopwatch,
                logger_=self.logger,
            )
        except Exception as e:
            self.logger.error("Error getting metafeatures (encoded) : %s", str(e))

        return res

    @classmethod
    def _default_smac_facade(
        cls,
        scenario: Scenario,
        target_function: AbstractRunner,
        metalearning_configurations,
        multi_objective_algorithm,
        # dask_client,
        callbacks,
        seed,
    ) -> AbstractFacade:
        """Creates the default optimizer, a specific SMAC object that is already
        configured well enough for us.

        Parameters
        ----------
        scenario : Scenario
        target_function : AbstractRunner
        metalearning_configurations
        dask_client
        multi_objective_algorithm
        multi_objective_kwargs

        Returns
        -------
        smac: AlgorithmConfigurationFacade
            The optimizer
        """
        # updatesmac: look into how intensification changed
        intensifier = Intensifier(scenario)

        encoder = RunHistoryLogEncoder(scenario=scenario, seed=seed)

        initial_design = None
        if len(metalearning_configurations) > 0:
            # TODO: The additional configs should be instead appended manually to
            # SMAC's runhistory, because otherwise SMAC will reevaluate these configs.
            initial_design = DefaultInitialDesign(
                scenario=scenario,
                n_configs=0,
                additional_configs=metalearning_configurations,
            )

        return AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=target_function,
            intensifier=intensifier,
            multi_objective_algorithm=multi_objective_algorithm,
            runhistory_encoder=encoder,
            initial_design=initial_design,
            callbacks=callbacks,
        )
