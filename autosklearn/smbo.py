import os
import time
import traceback

import numpy as np
import pandas as pd
import pynisher

# JTS TODO: notify aaron to clean up these nasty nested modules
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.util import impute_inactive_values

from smac.smbo.smbo import SMBO, get_types
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, \
    RunHistory2EPM4EIPS
from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.smbo.acquisition import EIPS
from smac.smbo import pSMAC

import autosklearn.metalearning
from autosklearn.constants import *
from autosklearn.metalearning.mismbo import suggest_via_metalearning
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.evaluation import eval_with_limits
from autosklearn.util import get_logger
from autosklearn.metalearning.metalearning.meta_base import MetaBase
from autosklearn.metalearning.metafeatures.metafeatures import \
    calculate_all_metafeatures_with_labels, calculate_all_metafeatures_encoded_labels

SENTINEL = 'uiaeo'

EXCLUDE_META_FEATURES_CLASSIFICATION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'PCA'
}

EXCLUDE_META_FEATURES_REGRESSION = {
    'Landmark1NN',
    'LandmarkDecisionNodeLearner',
    'LandmarkDecisionTree',
    'LandmarkLDA',
    'LandmarkNaiveBayes',
    'PCAFractionOfComponentsFor95PercentVariance',
    'PCAKurtosisFirstPC',
    'PCASkewnessFirstPC',
    'NumberOfClasses',
    'ClassOccurences',
    'ClassProbabilityMin',
    'ClassProbabilityMax',
    'ClassProbabilityMean',
    'ClassProbabilitySTD',
    'ClassEntropy',
    'LandmarkRandomNodeLearner',
    'PCA',
}


# dataset helpers
def load_data(dataset_info, backend):
    try:
        D = backend.load_datamanager()
    except IOError:
        D = None

    # Datamanager probably doesn't exist
    if D is None:
        D = CompetitionDataManager(dataset_info, encode_labels=True)
    return D


# metalearning helpers
def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            x_train, y_train, stopwatch, logger):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    stopwatch.start_task(task_name)
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if data_info_task in CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

    if data_info_task in [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION,
                          MULTILABEL_CLASSIFICATION, REGRESSION]:
        logger.info('Start calculating metafeatures for %s', basename)
        result = calculate_all_metafeatures_with_labels(
            x_train, y_train, categorical=categorical,
            dataset_name=basename+SENTINEL,
            dont_calculate=EXCLUDE_META_FEATURES, )
        for key in list(result.metafeature_values.keys()):
            if result.metafeature_values[key].type_ != 'METAFEATURE':
                del result.metafeature_values[key]

    else:
        result = None
        logger.info('Metafeatures not calculated')
    stopwatch.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (categorical attributes) took %5.2f',
        stopwatch.wall_elapsed(task_name))
    return result

def _calculate_metafeatures_encoded(basename, x_train, y_train, stopwatch,
                                    task, logger):
    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if task in CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

    task_name = 'CalculateMetafeaturesEncoded'
    stopwatch.start_task(task_name)
    result = calculate_all_metafeatures_encoded_labels(
        x_train, y_train, categorical=[False] * x_train.shape[1],
        dataset_name=basename+SENTINEL, dont_calculate=EXCLUDE_META_FEATURES)
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]
    stopwatch.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec',
        stopwatch.wall_elapsed(task_name))
    return result

def _get_metalearning_configurations(meta_base, basename, metric,
                                     configuration_space,
                                     task,
                                     initial_configurations_via_metalearning,
                                     is_sparse,
                                     stopwatch, logger):
    task_name = 'InitialConfigurations'
    stopwatch.start_task(task_name)
    try:
        metalearning_configurations = suggest_via_metalearning(
            meta_base, basename, metric,
            task,
            is_sparse == 1,
            initial_configurations_via_metalearning
        )
    except Exception as e:
        logger.error("Error getting metalearning configurations!")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        metalearning_configurations = []
    stopwatch.stop_task(task_name)
    return metalearning_configurations

def _print_debug_info_of_init_configuration(initial_configurations, basename,
                                            time_for_task, logger, stopwatch):
    logger.debug('Initial Configurations: (%d)' % len(initial_configurations))
    for initial_configuration in initial_configurations:
        logger.debug(initial_configuration)
    logger.debug('Looking for initial configurations took %5.2fsec',
                 stopwatch.wall_elapsed('InitialConfigurations'))
    logger.info(
        'Time left for %s after finding initial configurations: %5.2fsec',
        basename, time_for_task - stopwatch.wall_elapsed(basename))


class AutoMLScenario(Scenario):
    """
    We specialize the smac3 scenario here as we would like
    to create it in code, without actually reading a smac scenario file
    """

    def __init__(self, config_space, limit, cutoff_time, metafeatures,
                 output_dir, shared_model):
        self.logger = get_logger(self.__class__.__name__)

        # Give SMAC at least 5 seconds
        soft_limit = max(5, cutoff_time - 35)

        scenario_dict = {'cs': config_space,
                         'run_obj': 'quality',
                         'cutoff': soft_limit,
                         'algo_runs_timelimit': soft_limit,
                         'wallclock-limit': limit,
                         'features': metafeatures,
                         'instances': [[name] for name in metafeatures],
                         'output_dir': output_dir,
                         'shared_model': shared_model}

        super(AutoMLScenario, self).__init__(scenario_dict)


class OptimizationStrategy(object):
    def __init__(self, config_space,
                 datamanager,
                 backend,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 stopwatch,
                 metric,
                 logger,
                 seed=1):
        """Base class for optimization strategies in auto-sklearn

        Paramaters
        ----------
        config_space : ConfigSpace.ConfigurationSpace

        datamanager : autosklearn.data.AbstractDataManager

        backend : autosklearn.util.Backend

        total_walltime_limit : int

        func_eval_time_limit : int

        memory_limit : int

        stopwatch : autosklearn.util.StopWatch

        metric : autosklearn.metrics

        logger : logging.Logger

        seed : int
        """
        self.config_space = config_space
        self.datamanager = datamanager
        self.backend = backend
        self.total_walltime_limit = total_walltime_limit
        self.func_eval_time_limit = func_eval_time_limit
        self.memory_limit = memory_limit
        self.stopwatch = stopwatch
        self.metric = metric
        self.logger = logger
        self.seed = seed

    def get_metafeatures(self):
        raise NotImplementedError()

    def get_next_configurations(self, run_history,
                                meta_run_history, metafeatures):
        """Return a set of configurations to be evaluated

        Parameters
        ----------
        run_history : smac.run_history.RunHistory

        meta_run_history : smac.run_history.RunHistory
            RunHistory holding all information on runs performed on other
            datasets.

        metafeatures : dict

        Returns

        list

        """
        raise NotImplementedError()


class DefaultConfigurationsOptimizer(OptimizationStrategy):
    def get_metafeatures(self):
        return None

    def get_next_configurations(self, run_history,
                                meta_run_history, metafeatures):
        default_configs = []

        default_config = self.config_space.get_default_configuration()
        default_configs.append(default_config)

        if self.datamanager.info["task"] in CLASSIFICATION_TASKS:
            config_dict = {'balancing:strategy': 'weighting',
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

            config = Configuration(self.config_space, config_dict)
            default_configs.append(config)

        elif self.datamanager.info["task"] in REGRESSION_TASKS:
            config_dict = {'regressor:__choice__': 'sgd',
                           'regressor:sgd:loss': 'squared_loss',
                           'regressor:sgd:penalty': 'l2',
                           'regressor:sgd:alpha': 0.0001,
                           'regressor:sgd:fit_intercept': 'True',
                           'regressor:sgd:n_iter': 5,
                           'regressor:sgd:learning_rate': 'optimal',
                           'regressor:sgd:eta0': 0.01,
                           'regressor:sgd:average': 'True',
                           'imputation:strategy': 'mean',
                           'one_hot_encoding:use_minimum_fraction': 'True',
                           'one_hot_encoding:minimum_fraction': 0.1,
                           'preprocessor:__choice__': 'no_preprocessing',
                           'rescaling:__choice__': 'min/max'}

            config = Configuration(self.config_space, config_dict)
            default_configs.append(config)

        else:
            self.logger.info("Tasktype unknown: %s" %
                             TASK_TYPES_TO_STRING[self.datamanager.info[
                                 "task"]])

        return default_configs


class MetaLearningOptimizer(OptimizationStrategy):
    def __init__(self, *args, **kwargs):
        super(MetaLearningOptimizer, self).__init__(*args, **kwargs)
        self.metabase = None

    def collect_metalearning_suggestions(self, meta_base):
        metalearning_configurations = _get_metalearning_configurations(
            meta_base=meta_base,
            basename=self.datamanager.name + SENTINEL,
            metric=self.metric,
            configuration_space=self.config_space,
            task=self.datamanager.info['task'],
            is_sparse=self.datamanager.info['is_sparse'],
            initial_configurations_via_metalearning=self.num_metalearning_cfgs,
            stopwatch=self.stopwatch,
            logger=self.logger)

        _print_debug_info_of_init_configuration(
            metalearning_configurations,
            self.datamanager.data,
            self.total_walltime_limit,
            self.logger,
            self.stopwatch)

        return metalearning_configurations

    def _calculate_metafeatures(self):
        meta_features = _calculate_metafeatures(
            data_feat_type=self.datamanager.feat_type,
            data_info_task=self.datamanager.info['task'],
            x_train=self.datamanager.data['X_train'],
            y_train=self.datamanager.data['Y_train'],
            basename=self.datamanager.data,
            stopwatch=self.stopwatch,
            logger=self.logger)
        return meta_features

    def _calculate_metafeatures_with_limits(self, time_limit):
        res = None
        time_limit = max(time_limit, 1)
        try:
            safe_mf = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                              wall_time_in_s=int(time_limit),
                                              grace_period_in_s=30,
                                              logger=self.logger)(
                self._calculate_metafeatures)
            res = safe_mf()
        except Exception as e:
            self.logger.error('Error getting metafeatures: %s', str(e))

        return res

    def _calculate_metafeatures_encoded(self):
        meta_features_encoded = _calculate_metafeatures_encoded(
            self.datamanager.data,
            self.datamanager.data['X_train'],
            self.datamanager.data['Y_train'],
            self.stopwatch,
            self.datamanager.info['task'],
            self.logger)
        return meta_features_encoded

    def _calculate_metafeatures_encoded_with_limits(self, time_limit):
        res = None
        time_limit = max(time_limit, 1)
        try:
            safe_mf = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                              wall_time_in_s=int(time_limit),
                                              grace_period_in_s=30,
                                              logger=self.logger)(
                self._calculate_metafeatures_encoded)
            res = safe_mf()
        except Exception as e:
            self.logger.error('Error getting metafeatures (encoded) : %s',
                              str(e))

        return res

    def get_metafeatures(self):
        metafeature_calculation_time_limit = int(
            self.total_walltime_limit / 4)
        metafeature_calculation_start_time = time.time()
        meta_features = self._calculate_metafeatures_with_limits(
            metafeature_calculation_time_limit)
        metafeature_calculation_end_time = time.time()
        metafeature_calculation_time_limit = \
            metafeature_calculation_time_limit - (
                metafeature_calculation_end_time -
                metafeature_calculation_start_time)

        if metafeature_calculation_time_limit < 1:
            self.logger.warning('Time limit for metafeature calculation less '
                                'than 1 seconds (%f). Skipping calculation '
                                'of metafeatures for encoded dataset.',
                                metafeature_calculation_time_limit)
            meta_features_encoded = None
        else:
            self.datamanager.perform1HotEncoding()
            meta_features_encoded = \
                self._calculate_metafeatures_encoded_with_limits(
                    metafeature_calculation_time_limit)

        # In case there is a problem calculating the encoded meta-features
        if meta_features is None:
            if meta_features_encoded is not None:
                meta_features = meta_features_encoded
        else:
            if meta_features_encoded is not None:
                meta_features.metafeature_values.update(
                    meta_features_encoded.metafeature_values)

        if meta_features is not None:
            return meta_features
        else:
            return None

    def get_next_configurations(self,
                                run_history,
                                meta_run_history,
                                meta_features):
        metalearning_configurations = self.collect_metalearning_suggestions(
            meta_base)

        if metalearning_configurations is None:
            return None
        else:
            return metalearning_configurations

class SMACOptimizer(OptimizationStrategy):
    def __init__(self,
                 config_space,
                 datamanager,
                 backend,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 stopwatch,
                 metric,
                 logger,
                 seed=1,
                 shared_mode=False,
                 acquisition_function='EI'):
        super(SMACOptimizer, self).__init__(config_space,
                                            datamanager,
                                            backend,
                                            total_walltime_limit,
                                            func_eval_time_limit,
                                            memory_limit,
                                            stopwatch,
                                            metric,
                                            logger,
                                            seed=1)
        self.shared_mode = shared_mode
        self.acquisition_function = acquisition_function


    def get_metafeatures(self):
        return None

    def get_next_configurations(self, run_history,
                                meta_run_history, metafeatures):
        self.scenario = AutoMLScenario(self.config_space,
                                       self.total_walltime_limit,
                                       self.func_eval_time_limit,
                                       metafeatures,
                                       self.backend.temporary_directory,
                                       self.shared_mode)

        meta_features_list = []
        for meta_feature_name in metafeatures.columns:
            meta_features_list.append(metafeatures[meta_feature_name].value)
        meta_features_list = np.array(meta_features_list).reshape((1, -1))

        num_params = len(self.config_space.get_hyperparameters())
        types = get_types(self.config_space, self.scenario.feature_array)

        if self.acquisition_function == 'EI':
            rh2EPM = RunHistory2EPM4Cost(num_params=num_params,
                                         scenario=self.scenario,
                                         success_states=None,
                                         impute_censored_data=False,
                                         impute_state=None)
            model = RandomForestWithInstances(types,
                                              instance_features=meta_features_list,
                                              seed=1, num_trees=10)
            smac = SMBO(self.scenario, model=model, rng=self.seed)
        elif self.acquisition_function == 'EIPS':
            rh2EPM = RunHistory2EPM4EIPS(num_params=num_params,
                                         scenario=self.scenario,
                                         success_states=None,
                                         impute_censored_data=False,
                                         impute_state=None)
            model = UncorrelatedMultiObjectiveRandomForestWithInstances(
                ['cost', 'runtime'], types, num_trees=10,
                instance_features=meta_features_list, seed=1)
            acquisition_function = EIPS(model)
            smac = SMBO(self.scenario,
                        acquisition_function=acquisition_function,
                        model=model, runhistory2epm=rh2EPM, rng=self.seed)

        else:
            raise ValueError('Unknown acquisition function value %s!' %
                             self.acquisition_function)

        # Build a runtime model
        # runtime_rf = RandomForestWithInstances(types,
        #                                        instance_features=meta_features_list,
        #                                        seed=1, num_trees=10)
        # runtime_rh2EPM = RunHistory2EPM4EIPS(num_params=num_params,
        #                                      scenario=self.scenario,
        #                                      success_states=None,
        #                                      impute_censored_data=False,
        #                                      impute_state=None)
        # X_runtime, y_runtime = runtime_rh2EPM.transform(meta_runhistory)
        # runtime_rf.train(X_runtime, y_runtime[:, 1].flatten())
        # X_meta, Y_meta = rh2EPM.transform(meta_runhistory)
        # Transform Y_meta on a per-dataset base
        # for meta_dataset in meta_runs_dataset_indices:
        #     start_index, end_index = meta_runs_dataset_indices[meta_dataset]
        #     end_index += 1  # Python indexing
        #     Y_meta[start_index:end_index, 0] \
        #         [Y_meta[start_index:end_index, 0] > 2.0] = 2.0
        #     dataset_minimum = np.min(Y_meta[start_index:end_index, 0])
        #     Y_meta[start_index:end_index, 0] = 1 - (
        #         (1. - Y_meta[start_index:end_index, 0]) /
        #         (1. - dataset_minimum))
        #     Y_meta[start_index:end_index, 0] \
        #         [Y_meta[start_index:end_index, 0] > 2] = 2

        if self.scenario.shared_model:
            pSMAC.write(run_history=run_history,
                        output_directory=self.scenario.output_dir,
                        num_run=self.seed)
        if self.scenario.shared_model:
            pSMAC.read(run_history=run_history,
                       output_directory=self.scenario.output_dir,
                       configuration_space=self.config_space,
                       logger=self.logger)

        next_configs = []
        try:
            X_cfg, Y_cfg = rh2EPM.transform(run_history)

            if not run_history.empty():
                # Update costs by normalization
                dataset_minimum = np.min(Y_cfg[:, 0])
                Y_cfg[:, 0] = 1 - ((1. - Y_cfg[:, 0]) /
                                   (1. - dataset_minimum))
                Y_cfg[:, 0][Y_cfg[:, 0] > 2] = 2

            #if len(X_meta) > 0 and len(X_cfg) > 0:
            #    pass
            #    # X_cfg = np.concatenate((X_meta, X_cfg))
            #    # Y_cfg = np.concatenate((Y_meta, Y_cfg))
            #elif len(X_cfg) > 0:
            X_cfg = X_cfg.copy()
            Y_cfg = Y_cfg.copy()
            #else:
            #    raise ValueError('No training data for SMAC random forest!')

            self.logger.info('Using %d training points for SMAC.' %
                             X_cfg.shape[0])
            choose_next_start_time = time.time()
            next_configs_tmp = smac.choose_next(X_cfg, Y_cfg,
                                                num_interleaved_random=110,
                                                num_configurations_by_local_search=10,
                                                num_configurations_by_random_search_sorted=100)
            time_for_choose_next = time.time() - choose_next_start_time
            self.logger.info('Used %g seconds to find next '
                             'configurations' % (time_for_choose_next))
            next_configs.extend(next_configs_tmp)
        # TODO put Exception here!
        except Exception as e:
            self.logger.error(e)
            self.logger.error("Error in getting next configurations "
                              "with SMAC. Using random configuration!")
            next_config = self.config_space.sample_configuration()
            next_configs.append(next_config)

        return next_configs


class AutoMLSMBO(object):

    def __init__(self, config_space, dataset_name,
                 backend,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 stopwatch, start_num_run=1,
                 num_metalearning_cfgs=25,
                 smac_iters=1000,
                 seed=1,
                 metadata_directory=None,
                 resampling_strategy='holdout',
                 resampling_strategy_args=None,
                 acquisition_function='EI',
                 shared_mode=False):

        # data related
        self.dataset_name = dataset_name
        self.datamanager = None
        self.metric = None
        self.task = None
        self.backend = backend

        # the configuration space
        self.config_space = config_space

        # Evaluation
        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            resampling_strategy_args = {}
        self.resampling_strategy_args = resampling_strategy_args

        # and a bunch of useful limits
        self.total_walltime_limit = int(total_walltime_limit)
        self.func_eval_time_limit = int(func_eval_time_limit)
        self.memory_limit = memory_limit
        self.stopwatch = stopwatch
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.smac_iters = smac_iters
        self.start_num_run = start_num_run
        self.acquisition_function = acquisition_function
        self.shared_mode = shared_mode

        self.config_space.seed(self.seed)
        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed,
                                     ":" + dataset_name if dataset_name is
                                                           not None else "")
        self.logger = get_logger(logger_name)
        import logging
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

    def _load_metadata(self):
        task = self.datamanager.info['task']

        if self.metadata_directory is None:
            metalearning_directory = os.path.dirname(
                autosklearn.metalearning.__file__)
            # There is no multilabel data in OpenML
            if task == MULTILABEL_CLASSIFICATION:
                meta_task = BINARY_CLASSIFICATION
            else:
                meta_task = task
            metadata_directory = os.path.join(
                metalearning_directory, 'files',
                '%s_%s_%s' % (METRIC_TO_STRING[self.metric],
                              TASK_TYPES_TO_STRING[meta_task],
                              'sparse' if self.datamanager.info['is_sparse']
                              else 'dense'))
            self.metadata_directory = metadata_directory

        self.logger.info('Metadata directory: %s', self.metadata_directory)
        meta_runs_dataset_indices = {}
        meta_base = MetaBase(self.config_space, self.metadata_directory)
        meta_runhistory = RunHistory()

        meta_runs = meta_base.get_all_runs(METRIC_TO_STRING[self.metric])
        meta_runs_index = 0
        try:
            meta_durations = meta_base.get_all_runs('runtime')
            read_runtime_data = True
        except KeyError:
            read_runtime_data = False
            self.logger.critical('Cannot read runtime data.')
            if self.acquisition_function == 'EIPS':
                self.logger.critical(
                    'Reverting to acquisition function EI!')
                self.acquisition_function = 'EI'

        for meta_dataset in meta_runs.index:
            meta_dataset_start_index = meta_runs_index
            for meta_configuration in meta_runs.columns:
                if np.isfinite(
                        meta_runs.loc[meta_dataset, meta_configuration]):
                    try:
                        config = meta_base.get_configuration_from_algorithm_index(
                            meta_configuration)
                        cost = meta_runs.loc[
                            meta_dataset, meta_configuration]
                        if read_runtime_data:
                            runtime = meta_durations.loc[meta_dataset,
                                                         meta_configuration]
                        else:
                            runtime = 1
                        # TODO read out other status types!
                        meta_runhistory.add(config, cost, runtime,
                                            StatusType.SUCCESS,
                                            instance_id=meta_dataset)
                        meta_runs_index += 1
                    except:
                        # TODO maybe add warning
                        pass

            meta_runs_dataset_indices[meta_dataset] = (
                meta_dataset_start_index, meta_runs_index)

    def reset_data_manager(self):
        if self.datamanager is not None:
            del self.datamanager
        if isinstance(self.dataset_name, AbstractDataManager):
            self.datamanager = self.dataset_name
        else:
            self.datamanager = load_data(self.dataset_name,
                                         self.backend)
        self.metric = self.datamanager.info['metric']
        self.task = self.datamanager.info['task']

    def run_smbo(self):
        global evaluator

        # == first things first: load the datamanager
        self.reset_data_manager()
        
        # == Initialize SMBO stuff
        seed = self.seed # TODO

        run_history = RunHistory()
        meta_features = None
        num_run = self.start_num_run
        instance_id = self.dataset_name + SENTINEL

        incumbent = None
        # boolean variable determining whether to stop or not
        finished = False

        default_optimizer = DefaultConfigurationsOptimizer(self.config_space,
                                                           self.dataset_name,
                                                           self.backend,
                                                           self.total_walltime_limit,
                                                           self.func_eval_time_limit,
                                                           self.memory_limit,
                                                           self.stopwatch,
                                                           self.smac_iters,
                                                           self.seed,
                                                           self.logger)
        metalearning_optimizer = MetaLearningOptimizer(self.config_space,
                                                       self.dataset_name,
                                                       self.backend,
                                                       self.total_walltime_limit,
                                                       self.func_eval_time_limit,
                                                       self.memory_limit,
                                                       self.stopwatch,
                                                       self.smac_iters,
                                                       self.seed,
                                                       self.logger)
        smac_optimizer = SMACOptimizer(self.config_space,
                                       self.dataset_name,
                                       self.backend,
                                       self.total_walltime_limit,
                                       self.func_eval_time_limit,
                                       self.memory_limit,
                                       self.stopwatch,
                                       self.smac_iters,
                                       self.seed,
                                       self.logger,
                                       self.shared_mode)

        # Store the names, the number of configurations they are allowed to
        # run and the actual optimizer object
        optimizers = [('Default', 2, default_optimizer),
                      ('Meta-Learning', self.num_metalearning_cfgs,
                       metalearning_optimizer),
                      ('SMAC', self.smac_iters, smac_optimizer)]
        num_configs_for_optimizer = 0

        optimization_start_time = time.time()
        for optimizer_name, optimizer_configurations, optimizer in optimizers:
            while not finished:

                new_metafeatures = optimizer.get_metafeatures()
                if meta_features is not None and new_metafeatures is not None:
                    pass
                    # Do mean imputation of the meta-features - should be done specific
                    # for each prediction model!
                    meta_features_ = meta_features.fillna(metafeatures.mean(),
                                                          inplace=False)
                else:
                    meta_features_ = pd.DataFrame()

                _start_time = time.time()
                next_configs = optimizer.get_next_configurations()
                time_for_get_next_configs = time.time() - _start_time

                models_fitted_this_iteration = 0
                start_time_this_iteration = time.time()

                for next_config in next_configs:

                    # TODO subtract time if not enough time would be left...

                    self.logger.info("Starting to evaluate %d. configuration (from "
                                     "SMAC) with time limit %ds.", num_run,
                                     self.func_eval_time_limit)
                    self.logger.info(next_config)
                    self.reset_data_manager()
                    info = eval_with_limits(self.datamanager, self.backend, next_config,
                                            seed, num_run,
                                            self.resampling_strategy,
                                            self.resampling_strategy_args,
                                            self.memory_limit,
                                            self.func_eval_time_limit,
                                            logger=self.logger)
                    (duration, result, _, additional_run_info, status) = info
                    run_history.add(config=next_config, cost=result,
                                    time=duration, status=status,
                                    instance_id=instance_id, seed=seed)
                    run_history.update_cost(next_config, result)

                    # TODO add unittest to make sure everything works fine and
                    # this does not get outdated!
                    if incumbent is None:
                        incumbent = next_config
                    elif result < run_history.get_cost(incumbent):
                        incumbent = next_config

                    self.logger.info("Finished evaluating %d. configuration. "
                                     "Duration: %f; loss: %f; status %s; additional "
                                     "run info: %s ", num_run, duration, result,
                                     str(status), additional_run_info)
                    num_configs_for_optimizer += 1
                    num_run += 1

                    models_fitted_this_iteration += 1
                    time_used_this_iteration = time.time() - start_time_this_iteration

                    if time.time() - optimization_start_time <= 0:
                        finished = True
                        break

                    if models_fitted_this_iteration >= 2 and \
                            time_for_get_next_configs > 0 and \
                            time_used_this_iteration > time_for_get_next_configs:
                        break
                    elif time_for_get_next_configs <= 0 and \
                                    models_fitted_this_iteration >= 1:
                        break
                    elif models_fitted_this_iteration >= 50:
                        break
