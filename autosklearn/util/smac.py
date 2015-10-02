import os

import lockfile

import autosklearn.cli
from autosklearn.util.submit_process import submit_call
from autosklearn.util import logging_ as logging


def run_smac(tmp_dir, basename, time_for_task, ml_memory_limit,
              data_manager_path, configspace_path, initial_configurations,
              per_run_time_limit, watcher, backend, seed,
              resampling_strategy, resampling_strategy_arguments):
    logger = logging.get_logger(__name__)

    task_name = 'runSmac'
    watcher.start_task(task_name)

    instance_file_path, test_instance_file_path = \
        _write_instance_file(resampling_strategy, resampling_strategy_arguments,
                             data_manager_path, backend, tmp_dir)

    scenario_file_path = _write_scenario_file(time_for_task, per_run_time_limit,
                                              ml_memory_limit, tmp_dir,
                                              configspace_path,
                                              instance_file_path,
                                              test_instance_file_path,
                                              basename)

    # = Start SMAC
    time_smac = max(0, time_for_task - watcher.wall_elapsed(basename))
    logger.info('Start SMAC with %5.2fsec time left' % time_smac)

    initial_challengers = initial_configurations
    if initial_challengers is None:
        initial_challengers = []

    smac_options = {
        'retryTargetAlgorithmRunCount': '0',
        'intensification-percentage': '0.5',
        'num-ei-random': '1000',
        'num-challengers': 100,
        'initial-incumbent': 'DEFAULT',
        'validation': 'false',
    }

    call = ' '.join(['smac', '--numRun', str(seed), '--scenario',
                     scenario_file_path] +
                    ['--%s %s' % (opt, smac_options[opt])
                     for opt in smac_options]
                    + initial_challengers,
    )

    proc = submit_call(call, seed, logger)

    watcher.stop_task(task_name)
    return proc


def _write_instance_file(resampling_strategy, resampling_strategy_arguments,
                         data_manager_path, backend, tmp_dir):
    # = Create an instance file
    if resampling_strategy == 'holdout':
        instances = "holdout %s" % data_manager_path
    elif resampling_strategy == 'nested-cv':
        instances = "nested-cv:%d/%d %s" % (
            resampling_strategy_arguments['inner_folds'],
            resampling_strategy_arguments['outer_folds'],
            data_manager_path)
    elif resampling_strategy == 'cv':
        instances = "cv:%d %s" % (resampling_strategy_arguments['folds'],
                                   data_manager_path)
    elif resampling_strategy == 'partial-cv':
        instances = []
        folds = resampling_strategy_arguments['folds']
        for fold in range(folds):
            instances.append("partial-cv:%d/%d %s" % (fold, folds,
                                                       data_manager_path))
        instances = "\n".join(instances)
    else:
        raise ValueError(resampling_strategy)

    instance_file = os.path.join(tmp_dir, 'instances.txt')
    backend.write_txt_file(instance_file, instances, 'Instances')
    test_instance_file = os.path.join(tmp_dir, 'test_instances.txt')
    backend.write_txt_file(test_instance_file, 'test %s' % data_manager_path,
                           'Test instances')
    return instance_file, test_instance_file


def populate_argparse_with_resampling_arguments(parser):
    parser.add_argument("--resampling-strategy",
                        choices=["holdout", "cv", "partial-cv", "nested-cv"],
                        help="Resampling strategy used to estimate "
                             "generalization error.")
    parser.add_argument("--folds", type=int,
                        help="Number of cross-validation folds. Only used when "
                             "resampling strategy is either 'cv' or "
                             "'partial-cv'.")
    parser.add_argument("--outer-folds", type=int,
                        help="Number of outer cross-validation folds. Only used"
                             " when resampling strategy is nested-cv.")
    parser.add_argument("--inner-folds", type=int,
                        help="Number of inner cross-validation folds. Only used"
                             " when resampling strategy is nested-cv.")
    return parser


def namespace_to_automl_format(namespace):
    if namespace.resampling_strategy == 'holdout':
        strategy = 'holdout'
        arguments = None
    elif namespace.resampling_strategy == 'cv':
        strategy = 'cv'
        arguments = {'folds': namespace.folds}
    elif namespace.resampling_strategy == 'partial-cv':
        strategy = 'partial-cv'
        arguments = {'folds': namespace.folds}
    elif namespace.resampling_strategy == 'nested-cv':
        strategy = 'nested-cv'
        arguments = {'inner_folds': namespace.inner_folds,
                     'outer_folds': namespace.outer_folds}
    else:
        raise ValueError(namespace.resampling_strategy)
    return strategy, arguments


def _get_algo_exec(runsolver_limit, runsolver_delay, memory_limit):
    # Create call to autosklearn
    path_to_wrapper = os.path.dirname(
        os.path.abspath(autosklearn.cli.__file__))
    wrapper_exec = os.path.join(path_to_wrapper, 'SMAC_interface.py')
    call = 'python %s ' % wrapper_exec

    # Runsolver does strange things if the time limit is negative. Set it to
    # be at least one (0 means infinity)
    runsolver_limit = max(1, runsolver_limit)

    runsolver_prefix = 'runsolver --watcher-data /dev/null -W %d -d %d -M %d ' \
                       % (runsolver_limit, runsolver_delay, memory_limit)
    call = '"' + runsolver_prefix + ' ' + call + '"'
    return call


def _write_scenario_file(limit, cutoff_time, memory_limit, tmp_dir,
                         searchspace, instance_file_path,
                         test_instance_file_path, dataset_name):
    if limit <= 0:
    # It makes no sense to start building ensembles_statistics
        return

    limit = int(limit)
    wallclock_limit = int(limit)

    # It makes no sense to use less than 5sec
    # We try to do at least one run within the whole runtime
    runsolver_softlimit = max(5, cutoff_time - 35)
    runsolver_hardlimit_delay = 30

    algo_exec = _get_algo_exec(runsolver_softlimit, runsolver_hardlimit_delay,
                               memory_limit)

    scenario = {
        'cli-log-all-calls': 'false',
        'console-log-level': 'DEBUG',
        'log-level': 'DEBUG',
        'cutoffTime': str(runsolver_softlimit),
        'wallclock-limit': str(wallclock_limit),
        'intraInstanceObj': 'MEAN',
        'runObj': 'QUALITY',
        'algoExec': algo_exec,
        'numIterations': '2147483647',
        'totalNumRunsLimit': '2147483647',
        'outputDirectory': tmp_dir,
        'numConcurrentAlgoExecs': '1',
        'deterministic': 'true',
        'abort-on-first-run-crash': 'false',
        'pcs-file': os.path.abspath(searchspace),
        'execDir': tmp_dir,
        'transform-crashed-quality-value': '2',
        'instances': instance_file_path,
        'test-instances': test_instance_file_path
    }
    scenario_file = os.path.join(tmp_dir, '%s.scenario' % dataset_name)
    scenario_file_lock = scenario_file + '.lock'
    with lockfile.LockFile(scenario_file_lock):
        if not os.path.exists(scenario_file):
            with open(scenario_file, 'w') as fh:
                for option, value in scenario.items():
                    fh.write('%s = %s\n' % (option, value))

    return scenario_file
