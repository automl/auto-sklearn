import os
import time
import traceback
import warnings

import numpy as np
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
def load_data(dataset_info, backend, max_mem=None):
    try:
        D = backend.load_datamanager()
    except IOError:
        D = None

    # Datamanager probably doesn't exist
    if D is None:
        if max_mem is None:
            D = CompetitionDataManager(dataset_info, encode_labels=True)
        else:
            D = CompetitionDataManager(dataset_info, encode_labels=True, max_memory_in_mb=max_mem)
    return D


# metalearning helpers
def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            x_train, y_train, watcher, logger):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    watcher.start_task(task_name)
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
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (categorical attributes) took %5.2f',
        watcher.wall_elapsed(task_name))
    return result

def _calculate_metafeatures_encoded(basename, x_train, y_train, watcher,
                                    task, logger):
    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if task in CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

    task_name = 'CalculateMetafeaturesEncoded'
    watcher.start_task(task_name)
    result = calculate_all_metafeatures_encoded_labels(
        x_train, y_train, categorical=[False] * x_train.shape[1],
        dataset_name=basename+SENTINEL, dont_calculate=EXCLUDE_META_FEATURES)
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec',
        watcher.wall_elapsed(task_name))
    return result

def _get_metalearning_configurations(meta_base, basename, metric,
                                     configuration_space,
                                     task,
                                     initial_configurations_via_metalearning,
                                     is_sparse,
                                     watcher, logger):
    task_name = 'InitialConfigurations'
    watcher.start_task(task_name)
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
    watcher.stop_task(task_name)
    return metalearning_configurations

def _print_debug_info_of_init_configuration(initial_configurations, basename,
                                            time_for_task, logger, watcher):
    logger.debug('Initial Configurations: (%d)' % len(initial_configurations))
    for initial_configuration in initial_configurations:
        logger.debug(initial_configuration)
    logger.debug('Looking for initial configurations took %5.2fsec',
                 watcher.wall_elapsed('InitialConfigurations'))
    logger.info(
        'Time left for %s after finding initial configurations: %5.2fsec',
        basename, time_for_task - watcher.wall_elapsed(basename))


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
                         'run-obj': 'quality',
                         'cutoff-time': soft_limit,
                         'tuner-timeout': soft_limit,
                         'wallclock-limit': limit,
                         'features': metafeatures,
                         'instances': [[name] for name in metafeatures],
                         'output_dir': output_dir,
                         'shared_model': shared_model}

        super(AutoMLScenario, self).__init__(scenario_dict)
        # reset the logger, because otherwise we can't pickle the AutoMLScenario
        self.logger = get_logger(self.__class__.__name__)

class AutoMLSMBO(object):

    def __init__(self, config_space, dataset_name,
                 backend,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 watcher, start_num_run=1,
                 data_memory_limit=None,
                 num_metalearning_cfgs=25,
                 config_file=None,
                 smac_iters=1000,
                 seed=1,
                 metadata_directory=None,
                 resampling_strategy='holdout',
                 resampling_strategy_args=None,
                 acquisition_function='EI',
                 shared_mode=False):
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        #self.output_dir = output_dir
        #self.tmp_dir = tmp_dir
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
        self.data_memory_limit = data_memory_limit
        self.watcher = watcher
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.config_file = config_file
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.smac_iters = smac_iters
        self.start_num_run = start_num_run
        self.acquisition_function = acquisition_function
        self.shared_mode = shared_mode
        self.runhistory = None

        self.config_space.seed(self.seed)
        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed,
                                     ":" + dataset_name if dataset_name is
                                                           not None else "")
        self.logger = get_logger(logger_name)

    def _send_warnings_to_log(self, message, category, filename, lineno,
                              file=None):
        self.logger.debug('%s:%s: %s:%s', filename, lineno, category.__name__,
                          message)

    def reset_data_manager(self, max_mem=None):
        if max_mem is None:
            max_mem = self.data_memory_limit
        if self.datamanager is not None:
            del self.datamanager
        if isinstance(self.dataset_name, AbstractDataManager):
            self.datamanager = self.dataset_name
        else:
            self.datamanager = load_data(self.dataset_name,
                                         self.backend,
                                         max_mem = max_mem)
        self.metric = self.datamanager.info['metric']
        self.task = self.datamanager.info['task']
        
    def collect_additional_subset_defaults(self):
        default_configs = []
        # == set default configurations
        # first enqueue the default configuration from our config space
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
            try:
                config = Configuration(self.config_space, config_dict)
                default_configs.append(config)
            except ValueError as e:
                self.logger.warning("Second default configurations %s cannot"
                                    " be evaluated because of %s" %
                                    (config_dict, e))

            if self.datamanager.info["is_sparse"]:
                config_dict = {'classifier:__choice__': 'extra_trees',
                               'classifier:extra_trees:bootstrap': 'False',
                               'classifier:extra_trees:criterion': 'gini',
                               'classifier:extra_trees:max_depth': 'None',
                               'classifier:extra_trees:max_features': 1.0,
                               'classifier:extra_trees:min_samples_leaf': 5,
                               'classifier:extra_trees:min_samples_split': 5,
                               'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
                               'classifier:extra_trees:n_estimators': 100,
                               'balancing:strategy': 'weighting',
                               'imputation:strategy': 'mean',
                               'one_hot_encoding:use_minimum_fraction': 'True',
                               'one_hot_encoding:minimum_fraction': 0.1,
                               'preprocessor:__choice__': 'truncatedSVD',
                               'preprocessor:truncatedSVD:target_dim': 20,
                               'rescaling:__choice__': 'min/max'}
            else:
                n_data_points = self.datamanager.data['X_train'].shape[0]
                percentile = 20. / n_data_points
                percentile = max(percentile, 2.)

                config_dict = {'classifier:__choice__': 'extra_trees',
                               'classifier:extra_trees:bootstrap': 'False',
                               'classifier:extra_trees:criterion': 'gini',
                               'classifier:extra_trees:max_depth': 'None',
                               'classifier:extra_trees:max_features': 1.0,
                               'classifier:extra_trees:min_samples_leaf': 5,
                               'classifier:extra_trees:min_samples_split': 5,
                               'classifier:extra_trees:min_weight_fraction_leaf': 0.0,
                               'classifier:extra_trees:n_estimators': 100,
                               'balancing:strategy': 'weighting',
                               'imputation:strategy': 'mean',
                               'one_hot_encoding:use_minimum_fraction': 'True',
                               'one_hot_encoding:minimum_fraction': 0.1,
                               'preprocessor:__choice__': 'select_percentile_classification',
                               'preprocessor:select_percentile_classification:percentile': percentile,
                               'preprocessor:select_percentile_classification:score_func': 'chi2',
                               'rescaling:__choice__': 'min/max'}

            try:
                config = Configuration(self.config_space, config_dict)
                default_configs.append(config)
            except ValueError as e:
                self.logger.warning("Third default configurations %s cannot"
                                    " be evaluated because of %s" %
                                    (config_dict, e))

            if self.datamanager.info["is_sparse"]:
                config_dict = {'balancing:strategy': 'weighting',
                               'classifier:__choice__': 'multinomial_nb',
                               'classifier:multinomial_nb:alpha': 1.0,
                               'classifier:multinomial_nb:fit_prior': 'True',
                               'imputation:strategy': 'mean',
                               'one_hot_encoding:use_minimum_fraction': 'True',
                               'one_hot_encoding:minimum_fraction': 0.1,
                               'preprocessor:__choice__': 'no_preprocessing',
                               'rescaling:__choice__': 'none'}
            else:
                config_dict = {'balancing:strategy': 'weighting',
                               'classifier:__choice__': 'gaussian_nb',
                               'imputation:strategy': 'mean',
                               'one_hot_encoding:use_minimum_fraction': 'True',
                               'one_hot_encoding:minimum_fraction': 0.1,
                               'preprocessor:__choice__': 'no_preprocessing',
                               'rescaling:__choice__': 'standardize'}
            try:
                config = Configuration(self.config_space, config_dict)
                default_configs.append(config)
            except ValueError as e:
                self.logger.warning("Forth default configurations %s cannot"
                                    " be evaluated because of %s" %
                                    (config_dict, e))

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
            try:
                config = Configuration(self.config_space, config_dict)
                default_configs.append(config)
            except ValueError as e:
                self.logger.warning("Second default configurations %s cannot"
                                    " be evaluated because of %s" %
                                    (config_dict, e))

            if self.datamanager.info["is_sparse"]:
                config_dict = {'regressor:__choice__': 'extra_trees',
                               'regressor:extra_trees:bootstrap': 'False',
                               'regressor:extra_trees:criterion': 'mse',
                               'regressor:extra_trees:max_depth': 'None',
                               'regressor:extra_trees:max_features': 1.0,
                               'regressor:extra_trees:min_samples_leaf': 5,
                               'regressor:extra_trees:min_samples_split': 5,
                               'regressor:extra_trees:n_estimators': 100,
                               'imputation:strategy': 'mean',
                               'one_hot_encoding:use_minimum_fraction': 'True',
                               'one_hot_encoding:minimum_fraction': 0.1,
                               'preprocessor:__choice__': 'truncatedSVD',
                               'preprocessor:truncatedSVD:target_dim': 10,
                               'rescaling:__choice__': 'min/max'}
            else:
                config_dict = {'regressor:__choice__': 'extra_trees',
                               'regressor:extra_trees:bootstrap': 'False',
                               'regressor:extra_trees:criterion': 'mse',
                               'regressor:extra_trees:max_depth': 'None',
                               'regressor:extra_trees:max_features': 1.0,
                               'regressor:extra_trees:min_samples_leaf': 5,
                               'regressor:extra_trees:min_samples_split': 5,
                               'regressor:extra_trees:n_estimators': 100,
                               'imputation:strategy': 'mean',
                               'one_hot_encoding:use_minimum_fraction': 'True',
                               'one_hot_encoding:minimum_fraction': 0.1,
                               'preprocessor:__choice__': 'pca',
                               'preprocessor:pca:keep_variance': 0.9,
                               'preprocessor:pca:whiten': 'False',
                               'rescaling:__choice__': 'min/max'}

            try:
                config = Configuration(self.config_space, config_dict)
                default_configs.append(config)
            except ValueError as e:
                self.logger.warning("Third default configurations %s cannot"
                                    " be evaluated because of %s" %
                                    (config_dict, e))

        else:
            self.logger.info("Tasktype unknown: %s" %
                             TASK_TYPES_TO_STRING[self.datamanager.info[
                                 "task"]])

        return default_configs

    def collect_metalearning_suggestions(self, meta_base):
        metalearning_configurations = _get_metalearning_configurations(
            meta_base=meta_base,
            basename=self.dataset_name+SENTINEL,
            metric=self.metric,
            configuration_space=self.config_space,
            task=self.task,
            is_sparse=self.datamanager.info['is_sparse'],
            initial_configurations_via_metalearning=self.num_metalearning_cfgs,
            watcher=self.watcher,
            logger=self.logger)
        _print_debug_info_of_init_configuration(
            metalearning_configurations,
            self.dataset_name,
            self.total_walltime_limit,
            self.logger,
            self.watcher)

        return metalearning_configurations

    def _calculate_metafeatures(self):
        with warnings.catch_warnings():
            warnings.showwarning = self._send_warnings_to_log

            meta_features = _calculate_metafeatures(
                data_feat_type=self.datamanager.feat_type,
                data_info_task=self.datamanager.info['task'],
                x_train=self.datamanager.data['X_train'],
                y_train=self.datamanager.data['Y_train'],
                basename=self.dataset_name,
                watcher=self.watcher,
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
        with warnings.catch_warnings():
            warnings.showwarning = self._send_warnings_to_log

            meta_features_encoded = _calculate_metafeatures_encoded(
                self.dataset_name,
                self.datamanager.data['X_train'],
                self.datamanager.data['Y_train'],
                self.watcher,
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

    def run_smbo(self, max_iters=1000):
        global evaluator

        self.watcher.start_task('SMBO')

        # == first things first: load the datamanager
        self.reset_data_manager()
        
        # == Initialize SMBO stuff
        # first create a scenario
        seed = self.seed # TODO
        num_params = len(self.config_space.get_hyperparameters())
        # allocate a run history
        run_history = RunHistory()
        meta_runhistory = RunHistory()
        meta_runs_dataset_indices = {}
        num_run = self.start_num_run
        instance_id = self.dataset_name + SENTINEL

        # == Train on subset
        #    before doing anything, let us run the default_cfg
        #    on a subset of the available data to ensure that
        #    we at least have some models
        #    we will try three different ratios of decreasing magnitude
        #    in the hope that at least on the last one we will be able
        #    to get a model
        n_data = self.datamanager.data['X_train'].shape[0]
        subset_ratio = 10000. / n_data
        if subset_ratio >= 0.5:
            subset_ratio = 0.33
            subset_ratios = [subset_ratio, subset_ratio * 0.10]
        else:
            subset_ratios = [subset_ratio, 500. / n_data]
        self.logger.info("Training default configurations on a subset of "
                         "%d/%d data points." %
                         (int(n_data * subset_ratio), n_data))

        # the time limit for these function evaluations is rigorously
        # set to only 1/2 of a full function evaluation
        subset_time_limit = max(5, int(self.func_eval_time_limit / 2))
        # the configs we want to run on the data subset are:
        # 1) the default configs
        # 2) a set of configs we selected for training on a subset
        subset_configs = [self.config_space.get_default_configuration()] \
                          + self.collect_additional_subset_defaults()
        subset_config_succesful = [False] * len(subset_configs)
        for subset_config_id, next_config in enumerate(subset_configs):
            for i, ratio in enumerate(subset_ratios):
                self.reset_data_manager()
                n_data_subsample = int(n_data * ratio)

                # run the config, but throw away the result afterwards
                # since this cfg was evaluated only on a subset
                # and we don't want  to confuse SMAC
                self.logger.info("Starting to evaluate %d on SUBSET "
                                 "with size %d and time limit %ds.",
                                 num_run, n_data_subsample,
                                 subset_time_limit)
                self.logger.info(next_config)
                _info = eval_with_limits(
                    datamanager=self.datamanager, backend=self.backend,
                    config=next_config, seed=seed, num_run=num_run,
                    resampling_strategy=self.resampling_strategy,
                    resampling_strategy_args=self.resampling_strategy_args,
                    memory_limit=self.memory_limit,
                    func_eval_time_limit=subset_time_limit,
                    subsample=n_data_subsample,
                    logger=self.logger)
                (duration, result, _, additional_run_info, status) = _info
                self.logger.info("Finished evaluating %d. configuration on SUBSET. "
                                 "Duration %f; loss %f; status %s; additional run "
                                 "info: %s ", num_run, duration, result,
                                 str(status), additional_run_info)

                num_run += 1
                if i < len(subset_ratios) - 1:
                    if status != StatusType.SUCCESS:
                        # Do not increase num_run here, because we will try
                        # the same configuration with less data
                        self.logger.info("A CONFIG did not finish "
                                         " for subset ratio %f -> going smaller",
                                         ratio)
                        continue
                    else:
                        self.logger.info("Finished SUBSET training successfully"
                                         " with ratio %f", ratio)
                        subset_config_succesful[subset_config_id] = True
                        break
                else:
                    if status != StatusType.SUCCESS:
                        self.logger.info("A CONFIG did not finish "
                                         " for subset ratio %f.",
                                         ratio)
                        continue
                    else:
                        self.logger.info("Finished SUBSET training successfully"
                                         " with ratio %f", ratio)
                        subset_config_succesful[subset_config_id] = True
                        break

        # Use the first non-failing configuration from the subsets as the new
        #  default configuration -> this guards us against the random forest
        # failing on large, sparse datasets
        default_cfg = None
        for subset_config_id, next_config in enumerate(subset_configs):
            if subset_config_succesful[subset_config_id]:
                default_cfg = next_config
                break
        if default_cfg is None:
            default_cfg = self.config_space.get_default_configuration()

        # == METALEARNING suggestions
        # we start by evaluating the defaults on the full dataset again
        # and add the suggestions from metalearning behind it

        if self.metadata_directory is None:
            metalearning_directory = os.path.dirname(
                autosklearn.metalearning.__file__)
            # There is no multilabel data in OpenML
            if self.task == MULTILABEL_CLASSIFICATION:
                meta_task = BINARY_CLASSIFICATION
            else:
                meta_task = self.task
            metadata_directory = os.path.join(
                metalearning_directory, 'files',
                '%s_%s_%s' % (METRIC_TO_STRING[self.metric],
                              TASK_TYPES_TO_STRING[meta_task],
                              'sparse' if self.datamanager.info['is_sparse']
                              else 'dense'))
            self.metadata_directory = metadata_directory

        self.logger.info('Metadata directory: %s', self.metadata_directory)
        meta_base = MetaBase(self.config_space, self.metadata_directory)

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
            with warnings.catch_warnings():
                warnings.showwarning = self._send_warnings_to_log
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
            meta_base.add_dataset(instance_id, meta_features)
            # Do mean imputation of the meta-features - should be done specific
            # for each prediction model!
            all_metafeatures = meta_base.get_metafeatures(
                features=list(meta_features.keys()))
            all_metafeatures.fillna(all_metafeatures.mean(), inplace=True)

            with warnings.catch_warnings():
                warnings.showwarning = self._send_warnings_to_log
                metalearning_configurations = self.collect_metalearning_suggestions(
                    meta_base)
            if metalearning_configurations is None:
                metalearning_configurations = []
            self.reset_data_manager()

            self.logger.info('%s', meta_features)

            # Convert meta-features into a dictionary because the scenario
            # expects a dictionary
            meta_features_dict = {}
            for dataset, series in all_metafeatures.iterrows():
                meta_features_dict[dataset] = series.values
            meta_features_list = []
            for meta_feature_name in all_metafeatures.columns:
                meta_features_list.append(meta_features[meta_feature_name].value)
            meta_features_list = np.array(meta_features_list).reshape((1, -1))
            self.logger.info(list(meta_features_dict.keys()))

            meta_runs = meta_base.get_all_runs(METRIC_TO_STRING[self.metric])
            meta_runs_index = 0
            try:
                meta_durations = meta_base.get_all_runs('runtime')
                read_runtime_data = True
            except KeyError:
                read_runtime_data = False
                self.logger.critical('Cannot read runtime data.')
                if self.acquisition_function == 'EIPS':
                    self.logger.critical('Reverting to acquisition function EI!')
                    self.acquisition_function = 'EI'

            for meta_dataset in meta_runs.index:
                meta_dataset_start_index = meta_runs_index
                for meta_configuration in meta_runs.columns:
                    if np.isfinite(meta_runs.loc[meta_dataset, meta_configuration]):
                        try:
                            config = meta_base.get_configuration_from_algorithm_index(
                                meta_configuration)
                            cost = meta_runs.loc[meta_dataset, meta_configuration]
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
        else:
            if self.acquisition_function == 'EIPS':
                self.logger.critical('Reverting to acquisition function EI!')
                self.acquisition_function = 'EI'
            meta_features_list = []
            meta_features_dict = {}
            metalearning_configurations = []

        self.scenario = AutoMLScenario(config_space=self.config_space,
                                       limit=self.total_walltime_limit,
                                       cutoff_time=self.func_eval_time_limit,
                                       metafeatures=meta_features_dict,
                                       output_dir=self.backend.temporary_directory,
                                       shared_model=self.shared_mode)

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
            smac = SMBO(self.scenario, model=model,
                        rng=seed)
        elif self.acquisition_function == 'EIPS':
            rh2EPM = RunHistory2EPM4EIPS(num_params=num_params,
                                         scenario=self.scenario,
                                         success_states=None,
                                         impute_censored_data=False,
                                         impute_state=None)
            model = UncorrelatedMultiObjectiveRandomForestWithInstances(
                ['cost', 'runtime'], types, num_trees = 10,
                instance_features=meta_features_list, seed=1)
            acquisition_function = EIPS(model)
            smac = SMBO(self.scenario,
                        acquisition_function=acquisition_function,
                        model=model, runhistory2epm=rh2EPM, rng=seed)
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
        X_meta, Y_meta = rh2EPM.transform(meta_runhistory)
        # Transform Y_meta on a per-dataset base
        for meta_dataset in meta_runs_dataset_indices:
            start_index, end_index = meta_runs_dataset_indices[meta_dataset]
            end_index += 1  # Python indexing
            Y_meta[start_index:end_index, 0]\
                [Y_meta[start_index:end_index, 0] >2.0] =  2.0
            dataset_minimum = np.min(Y_meta[start_index:end_index, 0])
            Y_meta[start_index:end_index, 0] = 1 - (
                (1. - Y_meta[start_index:end_index, 0]) /
                (1. - dataset_minimum))
            Y_meta[start_index:end_index, 0]\
                  [Y_meta[start_index:end_index, 0] > 2] = 2

        # == first, evaluate all metelearning and default configurations
        finished = False
        for i, next_config in enumerate(([default_cfg] +
                                          metalearning_configurations)):
            # Do not evaluate default configurations more than once
            if i >= len([default_cfg]) and next_config in [default_cfg]:
                continue

            config_name = 'meta-learning' if i >= len([default_cfg]) \
                else 'default'

            self.logger.info("Starting to evaluate %d. configuration "
                             "(%s configuration) with time limit %ds.",
                             num_run, config_name, self.func_eval_time_limit)
            self.logger.info(next_config)
            self.reset_data_manager()
            info = eval_with_limits(datamanager=self.datamanager,
                                    backend=self.backend,
                                    config=next_config,
                                    seed=seed, num_run=num_run,
                                    resampling_strategy=self.resampling_strategy,
                                    resampling_strategy_args=self.resampling_strategy_args,
                                    memory_limit=self.memory_limit,
                                    func_eval_time_limit=self.func_eval_time_limit,
                                    logger=self.logger)
            (duration, result, _, additional_run_info, status) = info
            run_history.add(config=next_config, cost=result,
                            time=duration, status=status,
                            instance_id=instance_id, seed=seed,
                            additional_info=additional_run_info)
            run_history.update_cost(next_config, result)
            self.logger.info("Finished evaluating %d. configuration. "
                             "Duration %f; loss %f; status %s; additional run "
                             "info: %s ", num_run, duration, result,
                             str(status), additional_run_info)
            num_run += 1
            if smac.incumbent is None:
                smac.incumbent = next_config
            elif result < run_history.get_cost(smac.incumbent):
                smac.incumbent = next_config

            if self.scenario.shared_model:
                pSMAC.write(run_history=run_history,
                            output_directory=self.scenario.output_dir,
                            num_run=self.seed)

            if self.watcher.wall_elapsed(
                    'SMBO') > self.total_walltime_limit:
                finished = True

            if finished:
                break

        # == after metalearning run SMAC loop
        smac.runhistory = run_history
        smac_iter = 0
        while not finished:
            if self.scenario.shared_model:
                pSMAC.read(run_history=run_history,
                           output_directory=self.scenario.output_dir,
                           configuration_space=self.config_space,
                           logger=self.logger)

            next_configs = []
            time_for_choose_next = -1
            try:
                X_cfg, Y_cfg = rh2EPM.transform(run_history)

                if not run_history.empty():
                    # Update costs by normalization
                    dataset_minimum = np.min(Y_cfg[:, 0])
                    Y_cfg[:, 0] = 1 - ((1. - Y_cfg[:, 0]) /
                                       (1. - dataset_minimum))
                    Y_cfg[:, 0][Y_cfg[:, 0] > 2] = 2

                if len(X_meta) > 0 and len(X_cfg) > 0:
                    pass
                    #X_cfg = np.concatenate((X_meta, X_cfg))
                    #Y_cfg = np.concatenate((Y_meta, Y_cfg))
                elif len(X_meta) > 0:
                    X_cfg = X_meta.copy()
                    Y_cfg = Y_meta.copy()
                elif len(X_cfg) > 0:
                    X_cfg = X_cfg.copy()
                    Y_cfg = Y_cfg.copy()
                else:
                    raise ValueError('No training data for SMAC random forest!')

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

            models_fitted_this_iteration = 0
            start_time_this_iteration = time.time()
            for next_config in next_configs:
                x_runtime = impute_inactive_values(next_config)
                x_runtime = impute_inactive_values(x_runtime).get_array()
                # predicted_runtime = runtime_rf.predict_marginalized_over_instances(
                #     x_runtime.reshape((1, -1)))
                # predicted_runtime = np.exp(predicted_runtime[0][0][0]) - 1

                self.logger.info("Starting to evaluate %d. configuration (from "
                                 "SMAC) with time limit %ds.", num_run,
                                 self.func_eval_time_limit)
                self.logger.info(next_config)
                self.reset_data_manager()
                info = eval_with_limits(datamanager=self.datamanager,
                                        backend=self.backend,
                                        config=next_config,
                                        seed=seed, num_run=num_run,
                                        resampling_strategy=self.resampling_strategy,
                                        resampling_strategy_args=self.resampling_strategy_args,
                                        memory_limit=self.memory_limit,
                                        func_eval_time_limit=self.func_eval_time_limit,
                                        logger=self.logger)
                (duration, result, _, additional_run_info, status) = info
                run_history.add(config=next_config, cost=result,
                                time=duration, status=status,
                                instance_id=instance_id, seed=seed,
                                additional_info=additional_run_info)
                run_history.update_cost(next_config, result)

                #self.logger.info('Predicted runtime %g, true runtime %g',
                #                 predicted_runtime, duration)

                # TODO add unittest to make sure everything works fine and
                # this does not get outdated!
                if smac.incumbent is None:
                    smac.incumbent = next_config
                elif result < run_history.get_cost(smac.incumbent):
                    smac.incumbent = next_config

                self.logger.info("Finished evaluating %d. configuration. "
                                 "Duration: %f; loss: %f; status %s; additional "
                                 "run info: %s ", num_run, duration, result,
                                 str(status), additional_run_info)
                smac_iter += 1
                num_run += 1

                models_fitted_this_iteration += 1
                time_used_this_iteration = time.time() - start_time_this_iteration

                if max_iters is not None:
                    finished = (smac_iter >= max_iters)

                if self.watcher.wall_elapsed(
                        'SMBO') > self.total_walltime_limit:
                    finished = True

                if models_fitted_this_iteration >= 2 and \
                        time_for_choose_next > 0 and \
                        time_used_this_iteration > time_for_choose_next:
                    break
                elif time_for_choose_next <= 0 and \
                        models_fitted_this_iteration >= 1:
                    break
                elif models_fitted_this_iteration >= 50:
                    break

                if finished:
                    break

            if self.scenario.shared_model:
                pSMAC.write(run_history=run_history,
                            output_directory=self.scenario.output_dir,
                            num_run=self.seed)

        self.runhistory = run_history
        
        
