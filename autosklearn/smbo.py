import typing
from typing import Dict, List, Optional, Sequence

import copy
import json
import logging
import multiprocessing
import os
import time
import traceback
import warnings

import dask.distributed
import pynisher
from smac.callbacks import IncorporateRunResultCallback
from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.intensification import Intensifier
from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.optimizer.multi_objective.parego import ParEGO
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.serial_runner import SerialRunner

import autosklearn.metalearning
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    CLASSIFICATION_TASKS,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    TASK_TYPES_TO_STRING,
)
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.ensemble_building import EnsembleBuilderManager
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.metalearning.metafeatures.metafeatures import (
    calculate_all_metafeatures_encoded_labels,
    calculate_all_metafeatures_with_labels,
)
from autosklearn.metalearning.metalearning.meta_base import MetaBase
from autosklearn.metalearning.mismbo import suggest_via_metalearning
from autosklearn.metrics import Scorer
from autosklearn.util.logging_ import get_named_client_logger
from autosklearn.util.parallel import preload_modules
from autosklearn.util.stopwatch import StopWatch

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


def get_smac_object(
    scenario_dict,
    seed,
    ta,
    ta_kwargs,
    metalearning_configurations,
    n_jobs,
    dask_client,
    multi_objective_algorithm,
    multi_objective_kwargs,
):
    if len(scenario_dict["instances"]) > 1:
        intensifier = Intensifier
    else:
        intensifier = SimpleIntensifier

    scenario = Scenario(scenario_dict)
    if len(metalearning_configurations) > 0:
        default_config = scenario.cs.get_default_configuration()
        initial_configurations = [default_config] + metalearning_configurations
    else:
        initial_configurations = None
    rh2EPM = RunHistory2EPM4LogCost
    return SMAC4AC(
        scenario=scenario,
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=initial_configurations,
        run_id=seed,
        intensifier=intensifier,
        dask_client=dask_client,
        n_jobs=n_jobs,
        multi_objective_algorithm=multi_objective_algorithm,
        multi_objective_kwargs=multi_objective_kwargs,
    )


class AutoMLSMBO:
    def __init__(
        self,
        config_space,
        dataset_name,
        backend,
        total_walltime_limit,
        func_eval_time_limit,
        memory_limit,
        metrics: Sequence[Scorer],
        stopwatch: StopWatch,
        n_jobs,
        dask_client: dask.distributed.Client,
        port: int,
        start_num_run=1,
        data_memory_limit=None,
        num_metalearning_cfgs=25,
        config_file=None,
        seed=1,
        metadata_directory=None,
        resampling_strategy="holdout",
        resampling_strategy_args=None,
        include: Optional[Dict[str, List[str]]] = None,
        exclude: Optional[Dict[str, List[str]]] = None,
        disable_file_output=False,
        smac_scenario_args=None,
        get_smac_object_callback=None,
        scoring_functions=None,
        pynisher_context="spawn",
        ensemble_callback: typing.Optional[EnsembleBuilderManager] = None,
        trials_callback: typing.Optional[IncorporateRunResultCallback] = None,
    ):
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.datamanager = None
        self.metrics = metrics
        self.task = None
        self.backend = backend
        self.port = port

        # the configuration space
        self.config_space = config_space

        # the number of parallel workers/jobs
        self.n_jobs = n_jobs
        self.dask_client = dask_client

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = {}
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.worst_possible_result = get_cost_of_crash(self.metrics)
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.data_memory_limit = data_memory_limit
        self.stopwatch = stopwatch
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.config_file = config_file
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.start_num_run = start_num_run
        self.include = include
        self.exclude = exclude
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback
        self.scoring_functions = scoring_functions

        self.pynisher_context = pynisher_context

        self.ensemble_callback = ensemble_callback
        self.trials_callback = trials_callback

        dataset_name_ = "" if dataset_name is None else dataset_name
        logger_name = "%s(%d):%s" % (
            self.__class__.__name__,
            self.seed,
            ":" + dataset_name_,
        )
        if port is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = get_named_client_logger(
                name=logger_name,
                port=self.port,
            )

    def reset_data_manager(self, max_mem=None):
        if max_mem is None:
            max_mem = self.data_memory_limit
        if self.datamanager is not None:
            del self.datamanager
        if isinstance(self.dataset_name, AbstractDataManager):
            self.datamanager = self.dataset_name
        else:
            self.datamanager = self.backend.load_datamanager()

        self.task = self.datamanager.info["task"]

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
            safe_mf = pynisher.enforce_limits(
                mem_in_mb=self.memory_limit,
                wall_time_in_s=int(time_limit),
                grace_period_in_s=30,
                context=context,
                logger=self.logger,
            )(_calculate_metafeatures)
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
            safe_mf = pynisher.enforce_limits(
                mem_in_mb=self.memory_limit,
                wall_time_in_s=int(time_limit),
                grace_period_in_s=30,
                context=context,
                logger=self.logger,
            )(_calculate_metafeatures_encoded)
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

    def run_smbo(self):

        self.stopwatch.start("SMBO")

        # == first things first: load the datamanager
        self.reset_data_manager()

        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        # allocate a run history
        num_run = self.start_num_run

        # Initialize some SMAC dependencies

        metalearning_configurations = self.get_metalearning_suggestions()

        if self.resampling_strategy in ["partial-cv", "partial-cv-iterative-fit"]:
            num_folds = self.resampling_strategy_args["folds"]
            instances = [
                [json.dumps({"task_id": self.dataset_name, "fold": fold_number})]
                for fold_number in range(num_folds)
            ]
        else:
            instances = [[json.dumps({"task_id": self.dataset_name})]]

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over

        ta_kwargs = dict(
            backend=copy.deepcopy(self.backend),
            autosklearn_seed=seed,
            resampling_strategy=self.resampling_strategy,
            initial_num_run=num_run,
            include=self.include,
            exclude=self.exclude,
            metrics=self.metrics,
            memory_limit=self.memory_limit,
            disable_file_output=self.disable_file_output,
            scoring_functions=self.scoring_functions,
            port=self.port,
            pynisher_context=self.pynisher_context,
            **self.resampling_strategy_args,
        )
        ta = ExecuteTaFuncWithQueue

        startup_time = self.stopwatch.time_since(self.dataset_name, "start")
        total_walltime_limit = self.total_walltime_limit - startup_time - 5

        scenario_dict = {
            "abort_on_first_run_crash": False,
            "save-results-instantly": True,
            "cs": self.config_space,
            "cutoff_time": self.func_eval_time_limit,
            "deterministic": "true",
            "instances": instances,
            "memory_limit": self.memory_limit,
            "output-dir": self.backend.get_smac_output_directory(),
            "run_obj": "quality",
            "wallclock_limit": total_walltime_limit,
            "cost_for_crash": self.worst_possible_result,
        }
        if self.smac_scenario_args is not None:
            for arg in [
                "abort_on_first_run_crash",
                "cs",
                "deterministic",
                "instances",
                "output-dir",
                "run_obj",
                "shared-model",
                "cost_for_crash",
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        "Cannot override scenario argument %s, " "will ignore this.",
                        arg,
                    )
                    del self.smac_scenario_args[arg]
            for arg in [
                "cutoff_time",
                "memory_limit",
                "wallclock_limit",
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        "Overriding scenario argument %s: %s with value %s",
                        arg,
                        scenario_dict[arg],
                        self.smac_scenario_args[arg],
                    )
            scenario_dict.update(self.smac_scenario_args)

        smac_args = {
            "scenario_dict": scenario_dict,
            "seed": seed,
            "ta": ta,
            "ta_kwargs": ta_kwargs,
            "metalearning_configurations": metalearning_configurations,
            "n_jobs": self.n_jobs,
            "dask_client": self.dask_client,
        }
        if len(self.metrics) > 1:
            smac_args["multi_objective_algorithm"] = ParEGO
            smac_args["multi_objective_kwargs"] = {"rho": 0.05}
            scenario_dict["multi_objectives"] = [metric.name for metric in self.metrics]
        else:
            smac_args["multi_objective_algorithm"] = None
            smac_args["multi_objective_kwargs"] = {}
        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(**smac_args)
        else:
            smac = get_smac_object(**smac_args)

        if self.ensemble_callback is not None:
            smac.register_callback(self.ensemble_callback)
        if self.trials_callback is not None:
            smac.register_callback(self.trials_callback)

        smac.optimize()

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory
        if isinstance(smac.solver.tae_runner, DaskParallelRunner):
            self._budget_type = smac.solver.tae_runner.single_worker.budget_type
        elif isinstance(smac.solver.tae_runner, SerialRunner):
            self._budget_type = smac.solver.tae_runner.budget_type
        else:
            raise NotImplementedError(type(smac.solver.tae_runner))

        self.stopwatch.stop("SMBO")

        return self.runhistory, self.trajectory, self._budget_type

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
                    self.reset_data_manager()

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
