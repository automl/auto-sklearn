import os
import signal
import multiprocessing
import pynisher

import smac
# JTS TODO: notify aaron to clean up these nasty nested modules
from smac.smbo.smbo import SMBO
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM

from autosklearn.constants import *
from autosklearn.evaluation import HoldoutEvaluator
from autosklearn.metalearning.mismbo import \
    calc_meta_features, calc_meta_features_encoded, \
    suggest_via_metalearning
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.util import StopWatch, get_logger, setup_logger
from autosklearn.util import Backend

# dataset helpers
def load_data(dataset_info, outputdir, tmp_dir=None, max_mem=None):
    if tmp_dir is None:
        tmp_dir = outputdir
    backend = Backend(outputdir, tmp_dir)

    if max_mem is None:
        try:
            D = backend.load_datamanager()
        except IOError:
            D = None
    else:
        D = None

    # Datamanager probably doesn't exist
    if D is None:
        if max_mem is None:
            D = CompetitionDataManager(dataset_info, encode_labels=True)
        else:
            D = CompetitionDataManager(dataset_info, encode_labels=True, max_mem=max_mem)
    return D


# metalearning helpers
def _calculate_metafeatures(data_feat_type, data_info_task, basename,
                            metalearning_cnt, x_train, y_train, watcher,
                            logger):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    watcher.start_task(task_name)
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    if metalearning_cnt <= 0:
        result = None
    elif data_info_task in \
            [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION]:
        logger.info('Start calculating metafeatures for %s' % basename)
        result = calc_meta_features(x_train, y_train, categorical=categorical,
                                    dataset_name=basename)
    else:
        result = None
        logger.info('Metafeatures not calculated')
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (categorical attributes) took %5.2f' %
        watcher.wall_elapsed(task_name))
    return result


def _calculate_metafeatures_encoded(basename, x_train, y_train, watcher,
                                    logger):
    task_name = 'CalculateMetafeaturesEncoded'
    watcher.start_task(task_name)
    result = calc_meta_features_encoded(X_train=x_train, Y_train=y_train,
                                        categorical=[False] * x_train.shape[1],
                                        dataset_name=basename)
    watcher.stop_task(task_name)
    logger.info(
        'Calculating Metafeatures (encoded attributes) took %5.2fsec' %
        watcher.wall_elapsed(task_name))
    return result

def _get_metalearning_configurations(meta_features,
                                     meta_features_encoded, basename, metric,
                                     configuration_space,
                                     task, metadata_directory,
                                     initial_configurations_via_metalearning,
                                     is_sparse,
                                     watcher, logger):
    task_name = 'InitialConfigurations'
    watcher.start_task(task_name)
    try:
        metalearning_configurations = suggest_via_metalearning(
            meta_features,
            meta_features_encoded,
            configuration_space, basename, metric,
            task,
            is_sparse == 1,
            initial_configurations_via_metalearning,
            metadata_directory
        )
    except Exception as e:
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
    logger.debug('Looking for initial configurations took %5.2fsec' %
                 watcher.wall_elapsed('InitialConfigurations'))
    logger.info(
        'Time left for %s after finding initial configurations: %5.2fsec'
        % (basename, time_for_task - watcher.wall_elapsed(basename)))

# helpers for evaluating a configuration

evaluator = None

    
def _get_base_dict():
    return {
        'with_predictions': True,
        'all_scoring_functions': True,
        'output_y_test': True,
    }

# create closure for evaluating an algorithm
def _eval_config_and_save(configuration, data, tmp_dir, seed, num_run):
    global evaluator
    try:
        evaluator = HoldoutEvaluator(data, tmp_dir, configuration,
                                     seed=seed,
                                     num_run=num_run,
                                     **_get_base_dict())
        evaluator.fit()
        #signal.signal(15, empty_signal_handler)
        duration, result, seed, run_info = evaluator.finish_up()
        backend = Backend(None, tmp_dir)
        if os.path.exists(backend.get_model_dir()):
            backend.save_model(evaluator.model, num_run, seed)
        status = StatusType.SUCCESS
        return evaluator, (duration, result, seed, run_info, status)
    except:
        status = StatusType.CRASHED
        return evaluator, (None, None, seed, None, status)
    
    


def signal_handler(signum, frame):
    print('Received signal %s. Aborting Training!' % str(signum))
    global evaluator
    evaluator.finish_up()
    exit(0)

signal.signal(15, signal_handler)


class AutoMLScenario(Scenario):
    """
    We specialize the smac3 scenario here as we would like
    to create it in code, without actually reading a smac scenario file
    """

    def __init__(self, config_space, config_file, limit, cutoff_time, memory_limit, logger):
        self.logger = logger
        # we don't actually have a target algorithm here
        # we will implement algorithm calling and the SMBO loop ourselves
        self.ta = None
        self.execdir = None
        self.pcs_fn = os.path.abspath(config_file)
        self.run_obj = 'QUALITY'
        self.overall_obj = self.run_obj

        # Give SMAC at least 5 seconds
        soft_limit = max(5, cutoff_time - 35)
        self.cutoff = soft_limit
        self.algo_runs_timelimit = soft_limit
        self.wallclock_limit = limit

        # no instances
        self.train_inst_fn = None
        self.test_inst_fn = None
        self.feature_fn = None
        self.train_insts = []
        self.test_inst = []
        self.feature_dict = {}
        self.feature_array = None

        # save reference to config_space
        self.cs = config_space

class AutoMLSMBO(multiprocessing.Process):

    def __init__(self, config_space, dataset_name,
                 output_dir, tmp_dir,
                 limit, cutoff_time, memory_limit,
                 logger, watcher, start_num_run = 2,
                 default_cfgs = [],
                 num_metalearning_cfgs = 25,
                 config_file = None, smac_iters=1000,
                 seed = 1,
                 metadata_directory = None):
        super(AutoMLSMBO, self).__init__()
        # data related
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.datamanager = None
        self.metric = None
        self.task = None

        # the configuration space
        self.config_space = config_space

        # and a bunch of useful limits
        self.limit = limit
        self.cutoff_time = cutoff_time
        self.memory_limit = memory_limit
        self.logger = logger
        self.watcher = watcher
        self.default_cfgs = default_cfgs
        self.num_metalearning_cfgs = num_metalearning_cfgs
        self.config_file = config_file
        self.seed = seed
        self.metadata_directory = metadata_directory
        self.smac_iters = smac_iters
        self.start_num_run = start_num_run

    def reset_data_manager(self, max_mem=None):
        if self.datamanager is not None:
            del self.datamanager
        if isinstance(self.dataset_name, AbstractDataManager):
            self.datamanager = self.dataset_name
        else:
            self.datamanager = load_data(self.dataset_name,
                                         self.output_dir,
                                         self.tmp_dir,
                                         max_mem = max_mem)
        self.metric = self.datamanager.info['metric']
        self.task = self.datamanager.info['task']

    def collect_metalearning_suggestions(self):
        meta_features = _calculate_metafeatures(
            data_feat_type=self.datamanager.feat_type,
            data_info_task=self.datamanager.info['task'],
            x_train=self.datamanager.data['X_train'],
            y_train=self.datamanager.data['Y_train'],
            basename=self.dataset_name,
            watcher=self.watcher,
            metalearning_cnt=self.num_metalearning_cfgs,
            logger=self.logger)
        self.watcher.start_task('OneHot')
        self.datamanager.perform1HotEncoding()
        self.watcher.stop_task('OneHot')

        have_metafeatures = meta_features is not None
        known_task = self.datamanager.info['task'] in [MULTICLASS_CLASSIFICATION,
                                                       BINARY_CLASSIFICATION,
                                                       MULTILABEL_CLASSIFICATION]
        if have_metafeatures and known_task :
            meta_features_encoded = _calculate_metafeatures_encoded(
                self.dataset_name,
                self.datamanager.data['X_train'],
                self.datamanager.data['Y_train'],
                self.watcher,
                self.logger)

            metalearning_configurations = _get_metalearning_configurations(
                meta_features,
                meta_features_encoded,
                self.dataset_name,
                self.metric,
                self.config_space,
                self.task,
                self.metadata_directory,
                self.num_metalearning_cfgs,
                self.datamanager.info['is_sparse'],
                self.watcher,
                self.logger)
            print(metalearning_configurations)
            _print_debug_info_of_init_configuration(
                metalearning_configurations,
                self.dataset_name,
                self.limit,
                self.logger,
                self.watcher)

        else:
            metalearning_configurations = []
        return metalearning_configurations
        
        
    def collect_metalearning_suggestions_with_limits(self):
        res = None
        try:
            safe_suggest = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            cpu_time_in_s=int(self.scenario.cutoff),
                                            wall_time_in_s=int(self.scenario.wallclock_limit),
                                            grace_period_in_s=5)(self.collect_metalearning_suggestions)
            res = safe_suggest()
        except:
            pass
        if res is None:
            return []
        else:
            return res

    def run(self):
        # we use pynisher here to enforce limits
        safe_smbo = pynisher.enforce_limits(mem_in_mb=self.memory_limit,
                                            cpu_time_in_s=int(self.cutoff_time),
                                            wall_time_in_s=int(self.limit),
                                            grace_period_in_s=5)(self.run_smbo)
        safe_smbo(max_iters = self.smac_iters)
        

    def run_smbo(self, max_iters=1000):
        global evaluator

        # == first things first: load the datamanager
        self.reset_data_manager()
        
        # == Initialize SMBO stuff
        # first create a scenario
        seed = self.seed # TODO
        self.scenario = AutoMLScenario(self.config_space, self.config_file,
                                       self.limit, self.cutoff_time,
                                       self.memory_limit, self.logger)
        num_params = len(self.config_space.get_hyperparameters())
        # allocate a run history
        run_history = RunHistory()
        rh2EPM = RunHistory2EPM(num_params = num_params, cutoff_time = self.scenario.cutoff,
                                success_states = None, impute_censored_data = False,
                                impute_state = None)

        # == Train on subset
        #    before doing anything, let us run the default_cfgs
        #    on a subset of the available data to ensure that
        #    we at least have some models
        # TODO

        # == METALEARNING suggestions
        # we start by evaluating the defaults on the full dataset again
        # and add the suggestions from metalearning behind it
        metalearning_configurations = self.default_cfgs \
                                      + self.collect_metalearning_suggestions_with_limits()

        # == first evaluate all metelearning configurations
        num_run = self.start_num_run

        for config in metalearning_configurations:
            # JTS: reset the data manager before each configuration since
            #      we work on the data in-place
            # NOTE: this is where we could also apply some memory limits
            self.reset_data_manager()
            evaluator, info = _eval_config_and_save(config, self.datamanager,
                                                    self.tmp_dir, seed, num_run)
            (duration, result, _, additional_run_info, status) = info
            run_history.add(config = config, cost = result,
                            time = duration , status = status,
                            instance_id = 0, seed = seed)
            num_run += 1

        # == after metalearning run SMAC loop
        smac = SMBO(self.scenario, seed)
        smac_iter = 0
        finished = False
        while not finished:
            # JTS TODO: handle the case that run_history is empty
            X_cfg, Y_cfg = rh2EPM.transform(run_history)
            self.logger.debug("SMAC iteration : {}".format(smac_iter))
            next_config = smac.choose_next(X_cfg, Y_cfg)
            self.reset_data_manager()
            evaluator, info = _eval_config_and_save(next_config, self.datamanager,
                                                    self.tmp_dir, seed, num_run)
            (duration, result, _, additional_run_info, status) = info
            run_history.add(config = next_config, cost = result,
                            time = duration , status = status,
                            instance_id = 0, seed = seed)

            smac_iter += 1
            num_run += 1
            if max_iters is not None:
                finished = (smac_iter < max_iters)
        
        
