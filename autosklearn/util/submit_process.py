# -*- encoding: utf-8 -*-
from __future__ import print_function

import shlex
import subprocess
import os

import lockfile

import autosklearn.cli.SMAC_cli_holdout
from autosklearn.constants import *
from autosklearn.util.io import search_prog, search_prog_in_binaries
from conf import SCRIPT_ENSEMBLE_SELECTION


def get_prog_path(prog_name):
    prog_path = search_prog(prog_name)
    if prog_path is None:
        prog_path = search_prog_in_binaries(prog_name)
    assert prog_path is not None, "Not found %s binary file" % prog_name
    return os.path.abspath(prog_path)


def submit_call(call, seed, log_dir=None):
    print('Calling: ' + call)
    call = shlex.split(call)

    if log_dir is None:
        proc = subprocess.call(call, stdout=open(os.devnull, 'w'), shell=True)
    else:
        proc = subprocess.call(
            call,
            stdout=open(
                os.path.join(
                    log_dir,
                    'ensemble_out_%d.log' %
                    seed),
                'w'),
            stderr=open(os.path.join(log_dir, 'ensemble_err_%d.log' % seed),
                        'w'),
            shell=True)

    return proc


def get_algo_exec(runsolver_limit, runsolver_delay, memory_limit, *args):
    # Create call to autosklearn

    # todo
    # поменять местоположение этих скриптов и сделать нормальные названия
    # потому что эти скрипты вызывают при выполненинии
    # они не доступны при запуске без всей остальной штуки
    path_to_wrapper = os.path.dirname(
        os.path.abspath(autosklearn.cli.__file__))
    wrapper_exec = os.path.join(path_to_wrapper, 'SMAC_cli_holdout.py')
    call = 'python %s' % wrapper_exec

    # Runsolver does strange things if the time limit is negative. Set it to
    # be at least one (0 means infinity)
    runsolver_limit = max(1, runsolver_limit)

    runsolver_prefix = '%s --watcher-data /dev/null -W %d -d %d -M %d ' \
                       % (
                       get_prog_path('runsolver'), runsolver_limit, runsolver_delay,
                       memory_limit)
    call = '"' + runsolver_prefix + ' ' + call + ' ' + ' '.join(args) + '"'
    return call



def run_smac(dataset_name, dataset, tmp_dir, searchspace, instance_file, limit,
             cutoff_time, seed, memory_limit,
             initial_challengers=None, ):
    prog_name = "smac"

    if limit <= 0:
        # It makes no sense to start building ensembles_statistics
        return
    limit = int(limit)
    wallclock_limit = int(limit)

    # It makes no sense to use less than 5sec
    # We try to do at least one run within the whole runtime
    runsolver_softlimit = max(5, cutoff_time - 35)
    runsolver_hardlimit_delay = 30

    algo_exec = get_algo_exec(runsolver_softlimit, runsolver_hardlimit_delay,
                              memory_limit, dataset)

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
        'maxIncumbentRuns': '2147483647',
        'retryTargetAlgorithmRunCount': '0',
        'intensification-percentage': '0.5',
        'num-ei-random': '1000',
        # Number of challengers for local search
        'num-challengers': 100,
        'initial-incumbent': 'DEFAULT',
        'rf-split-min': '10',
        'validation': 'false',
        'deterministic': 'true',
        'abort-on-first-run-crash': 'false',
        'pcs-file': os.path.abspath(searchspace),
        'execDir': tmp_dir,
        'transform-crashed-quality-value': '2',
        'instances': instance_file
    }
    scenario_file = os.path.join(tmp_dir, '%s.scenario' % dataset_name)
    scenario_file_lock = scenario_file + '.lock'
    with lockfile.LockFile(scenario_file_lock):
        if not os.path.exists(scenario_file):
            with open(scenario_file, 'w') as fh:
                for option, value in scenario.items():
                    fh.write('%s = %s\n' % (option, value))

    if initial_challengers is None:
        initial_challengers = []

    call = ' '.join([get_prog_path('smac'), '--numRun', str(seed), '--scenario',
                     scenario_file] + initial_challengers)

    proc = submit_call(call, seed)
    return proc, call


def run_ensemble_builder(tmp_dir, dataset_name, task_type, metric, limit,
                         output_dir, ensemble_size, ensemble_nbest, seed,
                         ensemble_indices_output_dir):
    if limit <= 0:
        # It makes no sense to start building ensembles_statistics
        return
    wrapper_exec = SCRIPT_ENSEMBLE_SELECTION

    delay = 5

    task_type = TASK_TYPES_TO_STRING[task_type]

    call = ' '.join(['python', wrapper_exec, tmp_dir, dataset_name, task_type,
                     metric, str(limit - 5), output_dir, str(ensemble_size),
                     str(seed), ensemble_indices_output_dir])

    # Runsolver does strange things if the time limit is negative. Set it to
    # be at least one (0 means infinity)
    limit = max(1, limit)



    # Now add runsolver command
    # runsolver_cmd = "%s --watcher-data /dev/null -W %d" % \
    #                (runsolver_exec, limit)
    runsolver_cmd = '%s --watcher-data /dev/null -W %d -d %d' % \
                    (get_prog_path('runsolver'), limit, delay)
    call = runsolver_cmd + ' ' + call

    # print('Calling(debug): ' + call)
    # raise Exception("fsa")
    proc = submit_call(call, seed, log_dir=tmp_dir)
    return proc
