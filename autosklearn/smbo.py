import json
import os
import time
import traceback
import warnings

import numpy as np
import pynisher

from smac.facade.smac_facade import SMAC
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType


import autosklearn.metalearning
from autosklearn.constants import *
from autosklearn.metalearning.mismbo import suggest_via_metalearning
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.evaluation import ExecuteTaFuncWithQueue, WORST_POSSIBLE_RESULT
from autosklearn.util import get_logger
from autosklearn.metalearning.metalearning.meta_base import MetaBase
from autosklearn.metalearning.metafeatures.metafeatures import \
    calculate_all_metafeatures_with_labels, calculate_all_metafeatures_encoded_labels

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
            D = CompetitionDataManager(dataset_info)
        else:
            D = CompetitionDataManager(dataset_info, max_memory_in_mb=max_mem)
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
            dataset_name=basename,
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
        dataset_name=basename, dont_calculate=EXCLUDE_META_FEATURES)
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


def get_smac_object(
    scenario_dict,
    seed,
    ta,
    backend,
    metalearning_configurations,
    runhistory,
    run_id,
):
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario = Scenario(scenario_dict)
    if len(metalearning_configurations) > 0:
        default_config = scenario.cs.get_default_configuration()
        initial_configurations = [default_config] + metalearning_configurations
    else:
        initial_configurations = None
    rh2EPM = RunHistory2EPM4Cost(
        num_params=len(scenario.cs.get_hyperparameters()),
        scenario=scenario,
        success_states=[
            StatusType.SUCCESS,
            StatusType.MEMOUT,
            StatusType.TIMEOUT,
            # As long as we don't have a model for crashes yet!
            StatusType.CRASHED,
        ],
        impute_censored_data=False,
        impute_state=None,
    )
    return SMAC(
        scenario=scenario,
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        initial_configurations=initial_configurations,
        runhistory=runhistory,
        run_id=run_id,
    )


class AutoMLSMBO(object):

    def __init__(self, config_space, dataset_name,
                 backend,
                 total_walltime_limit,
                 func_eval_time_limit,
                 memory_limit,
                 metric,
                 watcher, start_num_run=1,
                 data_memory_limit=None,
                 num_metalearning_cfgs=25,
                 config_file=None,
                 seed=1,
                 metadata_directory=None,
                 resampling_strategy='holdout',
                 resampling_strategy_args=None,
                 shared_mode=False,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None,
                 disable_file_output=False,
                 smac_scenario_args=None,
                 get_smac_object_callback=None):
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.datamanager = None
        self.metric = metric
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
        self.start_num_run = start_num_run
        self.shared_mode = shared_mode
        self.include_estimators = include_estimators
        self.exclude_estimators = exclude_estimators
        self.include_preprocessors = include_preprocessors
        self.exclude_preprocessors = exclude_preprocessors
        self.disable_file_output = disable_file_output
        self.smac_scenario_args = smac_scenario_args
        self.get_smac_object_callback = get_smac_object_callback

        logger_name = '%s(%d):%s' % (self.__class__.__name__, self.seed,
                                     ":" + dataset_name if dataset_name is
                                                           not None else "")
        self.logger = get_logger(logger_name)

    def _send_warnings_to_log(self, message, category, filename, lineno,
                              file=None, line=None):
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
                                         max_mem=max_mem)

        self.task = self.datamanager.info['task']

    def collect_metalearning_suggestions(self, meta_base):
        metalearning_configurations = _get_metalearning_configurations(
            meta_base=meta_base,
            basename=self.dataset_name,
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

    def run_smbo(self):

        self.watcher.start_task('SMBO')

        # == first things first: load the datamanager
        self.reset_data_manager()
        
        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        num_params = len(self.config_space.get_hyperparameters())
        # allocate a run history
        num_run = self.start_num_run

        # Initialize some SMAC dependencies

        metalearning_configurations = self.get_metalearning_suggestions()

        if self.resampling_strategy in ['partial-cv',
                                        'partial-cv-iterative-fit']:
            num_folds = self.resampling_strategy_args['folds']
            instances = [[json.dumps({'task_id': self.dataset_name,
                                      'fold': fold_number})]
                         for fold_number in range(num_folds)]
        else:
            instances = [[json.dumps({'task_id': self.dataset_name})]]

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over
        exclude = dict()
        include = dict()
        if self.include_preprocessors is not None and \
                        self.exclude_preprocessors is not None:
            raise ValueError('Cannot specify include_preprocessors and '
                             'exclude_preprocessors.')
        elif self.include_preprocessors is not None:
            include['preprocessor'] = self.include_preprocessors
        elif self.exclude_preprocessors is not None:
            exclude['preprocessor'] = self.exclude_preprocessors

        if self.include_estimators is not None and \
                        self.exclude_estimators is not None:
            raise ValueError('Cannot specify include_estimators and '
                             'exclude_estimators.')
        elif self.include_estimators is not None:
            if self.task in CLASSIFICATION_TASKS:
                include['classifier'] = self.include_estimators
            elif self.task in REGRESSION_TASKS:
                include['regressor'] = self.include_estimators
            else:
                raise ValueError(self.task)
        elif self.exclude_estimators is not None:
            if self.task in CLASSIFICATION_TASKS:
                exclude['classifier'] = self.exclude_estimators
            elif self.task in REGRESSION_TASKS:
                exclude['regressor'] = self.exclude_estimators
            else:
                raise ValueError(self.task)

        ta = ExecuteTaFuncWithQueue(backend=self.backend,
                                    autosklearn_seed=seed,
                                    resampling_strategy=self.resampling_strategy,
                                    initial_num_run=num_run,
                                    logger=self.logger,
                                    include=include,
                                    exclude=exclude,
                                    metric=self.metric,
                                    memory_limit=self.memory_limit,
                                    disable_file_output=self.disable_file_output,
                                    **self.resampling_strategy_args)

        startup_time = self.watcher.wall_elapsed(self.dataset_name)
        total_walltime_limit = self.total_walltime_limit - startup_time - 5
        scenario_dict = {
            'abort_on_first_run_crash': False,
            'cs': self.config_space,
            'cutoff_time': self.func_eval_time_limit,
            'deterministic': 'true',
            'instances': instances,
            'memory_limit': self.memory_limit,
            'output-dir':
                self.backend.get_smac_output_directory(),
            'run_obj': 'quality',
            'shared-model': self.shared_mode,
            'wallclock_limit': total_walltime_limit,
            'cost_for_crash': WORST_POSSIBLE_RESULT,
        }
        if self.smac_scenario_args is not None:
            for arg in [
                'abort_on_first_run_crash',
                'cs',
                'deterministic',
                'instances',
                'output-dir',
                'run_obj',
                'shared-model',
                'cost_for_crash',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning('Cannot override scenario argument %s, '
                                        'will ignore this.', arg)
                    del self.smac_scenario_args[arg]
            for arg in [
                'cutoff_time',
                'memory_limit',
                'wallclock_limit',
            ]:
                if arg in self.smac_scenario_args:
                    self.logger.warning(
                        'Overriding scenario argument %s: %s with value %s',
                        arg,
                        scenario_dict[arg],
                        self.smac_scenario_args[arg]
                    )
            scenario_dict.update(self.smac_scenario_args)

        runhistory = RunHistory(aggregate_func=average_cost)
        smac_args = {
            'scenario_dict': scenario_dict,
            'seed': seed,
            'ta': ta,
            'backend': self.backend,
            'metalearning_configurations': metalearning_configurations,
            'runhistory': runhistory,
            'run_id': seed,
        }
        if self.get_smac_object_callback is not None:
            smac = self.get_smac_object_callback(**smac_args)
        else:
            smac = get_smac_object(**smac_args)

        smac.optimize()

        self.runhistory = smac.solver.runhistory
        self.trajectory = smac.solver.intensifier.traj_logger.trajectory

        return self.runhistory, self.trajectory


    def get_metalearning_suggestions(self):
        # == METALEARNING suggestions
        # we start by evaluating the defaults on the full dataset again
        # and add the suggestions from metalearning behind it
        if self.num_metalearning_cfgs > 0:
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
                    '%s_%s_%s' % (self.metric, TASK_TYPES_TO_STRING[meta_task],
                                  'sparse' if self.datamanager.info['is_sparse']
                                  else 'dense'))
                self.metadata_directory = metadata_directory

            if os.path.exists(self.metadata_directory):

                self.logger.info('Metadata directory: %s',
                                 self.metadata_directory)
                meta_base = MetaBase(self.config_space, self.metadata_directory)

                try:
                    meta_base.remove_dataset(self.dataset_name)
                except:
                    pass

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
                    self.logger.warning(
                        'Time limit for metafeature calculation less '
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
                    meta_base.add_dataset(self.dataset_name, meta_features)
                    # Do mean imputation of the meta-features - should be done specific
                    # for each prediction model!
                    all_metafeatures = meta_base.get_metafeatures(
                        features=list(meta_features.keys()))
                    all_metafeatures.fillna(all_metafeatures.mean(),
                                            inplace=True)

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
                        meta_features_list.append(
                            meta_features[meta_feature_name].value)
                    meta_features_list = np.array(meta_features_list).reshape(
                        (1, -1))
                    self.logger.info(list(meta_features_dict.keys()))

            else:
                meta_features = None
                self.logger.warning('Could not find meta-data directory %s' %
                                    metadata_directory)

        else:
            meta_features = None
        if meta_features is None:
            meta_features_list = []
            metalearning_configurations = []
        return metalearning_configurations
