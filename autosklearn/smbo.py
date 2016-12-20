import os
import time
import traceback
import warnings

import numpy as np
import pynisher

# JTS TODO: notify aaron to clean up these nasty nested modules
from ConfigSpace.configuration_space import Configuration

from smac.facade.smac_facade import SMAC
from smac.utils.util_funcs import get_types
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.smbo.objective import average_cost
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
from autosklearn.evaluation import ExecuteTaFuncWithQueue
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
        
        # == Initialize non-SMBO stuff
        # first create a scenario
        seed = self.seed
        self.config_space.seed(seed)
        num_params = len(self.config_space.get_hyperparameters())
        # allocate a run history
        num_run = self.start_num_run
        instance_id = self.dataset_name + SENTINEL

        # Initialize some SMAC dependencies
        runhistory = RunHistory(aggregate_func=average_cost)
        # meta_runhistory = RunHistory(aggregate_func=average_cost)
        # meta_runs_dataset_indices = {}

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

            #meta_runs = meta_base.get_all_runs(METRIC_TO_STRING[self.metric])
            #meta_runs_index = 0
            #try:
            #    meta_durations = meta_base.get_all_runs('runtime')
            #    read_runtime_data = True
            #except KeyError:
            #    read_runtime_data = False
            #    self.logger.critical('Cannot read runtime data.')
            #    if self.acquisition_function == 'EIPS':
            #        self.logger.critical('Reverting to acquisition function EI!')
            #        self.acquisition_function = 'EI'

            # for meta_dataset in meta_runs.index:
            #     meta_dataset_start_index = meta_runs_index
            #     for meta_configuration in meta_runs.columns:
            #         if np.isfinite(meta_runs.loc[meta_dataset, meta_configuration]):
            #             try:
            #                 config = meta_base.get_configuration_from_algorithm_index(
            #                     meta_configuration)
            #                 cost = meta_runs.loc[meta_dataset, meta_configuration]
            #                 if read_runtime_data:
            #                     runtime = meta_durations.loc[meta_dataset,
            #                                                  meta_configuration]
            #                 else:
            #                     runtime = 1
            #                 # TODO read out other status types!
            #                 meta_runhistory.add(config, cost, runtime,
            #                                     StatusType.SUCCESS,
            #                                     instance_id=meta_dataset)
            #                 meta_runs_index += 1
            #             except:
            #                 # TODO maybe add warning
            #                 pass
            #
            #     meta_runs_dataset_indices[meta_dataset] = (
            #         meta_dataset_start_index, meta_runs_index)
        else:
            if self.acquisition_function == 'EIPS':
                self.logger.critical('Reverting to acquisition function EI!')
                self.acquisition_function = 'EI'
            meta_features_list = []
            meta_features_dict = {}
            metalearning_configurations = []

        self.scenario = Scenario({'cs': self.config_space,
                                  'cutoff-time': self.func_eval_time_limit,
                                  'memory-limit': self.memory_limit,
                                  'wallclock-limit': self.total_walltime_limit,
                                  #'instances': [[name] for name in meta_features_dict],
                                  'output-dir': self.backend.temporary_directory,
                                  'shared-model': self.shared_mode,
                                  'run-obj': 'quality',
                                  'deterministic': 'true'})

        # TODO rebuild target algorithm to be it's own target algorithm
        # evaluator, which takes into account that a run can be killed prior
        # to the model being fully fitted; thus putting intermediate results
        # into a queue and querying them once the time is over
        ta = ExecuteTaFuncWithQueue(backend=self.backend,
                                    autosklearn_seed=seed,
                                    resampling_strategy=self.resampling_strategy,
                                    initial_num_run=num_run,
                                    logger=self.logger,
                                    **self.resampling_strategy_args)

        types = get_types(self.config_space, self.scenario.feature_array)

        # TODO extract generation of SMAC object into it's own function for
        # testing
        if self.acquisition_function == 'EI':
            model = RandomForestWithInstances(types,
                                              #instance_features=meta_features_list,
                                              seed=1, num_trees=10)
            rh2EPM = RunHistory2EPM4Cost(num_params=num_params,
                                         scenario=self.scenario,
                                         success_states=[StatusType.SUCCESS,
                                                         StatusType.MEMOUT,
                                                         StatusType.TIMEOUT],
                                         impute_censored_data=False,
                                         impute_state=None)
            smac = SMAC(scenario=self.scenario,
                        model=model,
                        rng=seed,
                        runhistory2epm=rh2EPM,
                        tae_runner=ta,
                        runhistory=runhistory)
        elif self.acquisition_function == 'EIPS':
            rh2EPM = RunHistory2EPM4EIPS(num_params=num_params,
                                         scenario=self.scenario,
                                         success_states=[StatusType.SUCCESS,
                                                         StatusType.MEMOUT,
                                                         StatusType.TIMEOUT],
                                         impute_censored_data=False,
                                         impute_state=None)
            model = UncorrelatedMultiObjectiveRandomForestWithInstances(
                ['cost', 'runtime'], types, num_trees = 10,
                instance_features=meta_features_list, seed=1)
            acquisition_function = EIPS(model)
            smac = SMAC(scenario=self.scenario, tae_runner=ta,
                        acquisition_function=acquisition_function,
                        model=model, runhistory2epm=rh2EPM, rng=seed,
                        runhistory=runhistory)
        else:
            raise ValueError('Unknown acquisition function value %s!' %
                             self.acquisition_function)

        smac.solver.stats.start_timing()
        smac.solver.incumbent = smac.solver.initial_design.run()

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
        # # Transform Y_meta on a per-dataset base
        # for meta_dataset in meta_runs_dataset_indices:
        #     start_index, end_index = meta_runs_dataset_indices[meta_dataset]
        #     end_index += 1  # Python indexing
        #     Y_meta[start_index:end_index, 0]\
        #         [Y_meta[start_index:end_index, 0] >2.0] =  2.0
        #     dataset_minimum = np.min(Y_meta[start_index:end_index, 0])
        #     Y_meta[start_index:end_index, 0] = 1 - (
        #         (1. - Y_meta[start_index:end_index, 0]) /
        #         (1. - dataset_minimum))
        #     Y_meta[start_index:end_index, 0]\
        #           [Y_meta[start_index:end_index, 0] > 2] = 2

        smac.solver.stats.start_timing()
        # == first, evaluate all metelearning and default configurations
        smac.solver.incumbent = smac.solver.initial_design.run()

        for challenger in metalearning_configurations:

            smac.solver.incumbent, inc_perf = smac.solver.intensifier.intensify(
                challengers=[challenger],
                incumbent=smac.solver.incumbent,
                run_history=smac.solver.runhistory,
                aggregate_func=smac.solver.aggregate_func,
                time_bound=self.total_walltime_limit)

            if smac.solver.scenario.shared_model:
                pSMAC.write(run_history=smac.solver.runhistory,
                            output_directory=smac.solver.scenario.output_dir,
                            num_run=self.seed)

            if smac.solver.stats.is_budget_exhausted():
                break

        # == after metalearning run SMAC loop
        while True:
            if smac.solver.scenario.shared_model:
                pSMAC.read(run_history=smac.solver.runhistory,
                           output_directory=self.scenario.output_dir,
                           configuration_space=self.config_space,
                           logger=self.logger)

            choose_next_start_time = time.time()
            try:
                challengers = self.choose_next(smac)
            except Exception as e:
                self.logger.error(e)
                self.logger.error("Error in getting next configurations "
                                  "with SMAC. Using random configuration!")
                next_config = self.config_space.sample_configuration()
                challengers = [next_config]
            time_for_choose_next = time.time() - choose_next_start_time
            self.logger.info('Used %g seconds to find next '
                             'configurations' % (time_for_choose_next))

            smac.solver.incumbent, inc_perf = smac.solver.intensifier.intensify(
                challengers=challengers,
                incumbent=smac.solver.incumbent,
                run_history=smac.solver.runhistory,
                aggregate_func=smac.solver.aggregate_func,
                time_bound=time_for_choose_next)

            if smac.solver.scenario.shared_model:
                pSMAC.write(run_history=smac.solver.runhistory,
                            output_directory=smac.solver.scenario.output_dir,
                            num_run=self.seed)

            if smac.solver.stats.is_budget_exhausted():
                break

        self.runhistory = smac.solver.runhistory
        return runhistory

    def choose_next(self, smac):
        challengers = []

        if len(smac.solver.runhistory.data) == 0:
            raise ValueError('Cannot use SMBO algorithm on empty runhistory.')

        X_cfg, Y_cfg = smac.solver.rh2EPM.transform(smac.solver.runhistory)

        if not smac.solver.runhistory.empty():
            # Update costs by normalization
            dataset_minimum = np.min(Y_cfg[:, 0])
            Y_cfg[:, 0] = 1 - ((1. - Y_cfg[:, 0]) /
                               (1. - dataset_minimum))
            Y_cfg[:, 0][Y_cfg[:, 0] > 2] = 2

        # if len(X_meta) > 0 and len(X_cfg) > 0:
        #    pass
        #    X_cfg = np.concatenate((X_meta, X_cfg))
        #    Y_cfg = np.concatenate((Y_meta, Y_cfg))
        # elif len(X_meta) > 0:
        #    X_cfg = X_meta.copy()
        #    Y_cfg = Y_meta.copy()
        # elif len(X_cfg) > 0:
        X_cfg = X_cfg.copy()
        Y_cfg = Y_cfg.copy()
        # else:
        #    raise ValueError('No training data for SMAC random forest!')

        self.logger.info('Using %d training points for SMAC.' %
                         X_cfg.shape[0])
        next_configs_tmp = smac.solver.choose_next(
            X_cfg, Y_cfg,
            num_configurations_by_local_search=10,
            num_configurations_by_random_search_sorted=100)

        challengers.extend(next_configs_tmp)

        return challengers

