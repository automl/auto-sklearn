# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import shlex
import subprocess

import psutil

from autosklearn.constants import *
from autosklearn.util import logging_ as logging


def submit_call(call, seed, logger, log_dir=None):
    logger.info('Calling: ' + call)
    call = shlex.split(call)

    if log_dir is None:
        try:
            proc = subprocess.Popen(call, stdout=open(os.devnull, 'w'))
        except OSError as e:
            logger.critical(e)
            logger.critical('Problem starting subprocess, see error message '
                            'above. PATH is %s' % os.environ['PATH'])
    else:
        try:
            proc = subprocess.Popen(
                call,
                stdout=open(
                    os.path.join(log_dir, 'ensemble_out_%d.log' % seed), 'w'),
                stderr=open(
                    os.path.join(log_dir, 'ensemble_err_%d.log' % seed), 'w'))
        except OSError as e:
            logger.critical(e)
            logger.critical('Problem starting subprocess, see error message '
                            'above. PATH is %s' % os.environ['PATH'])

    pid = proc.pid
    process = psutil.Process(pid)
    return process


def run_ensemble_builder(tmp_dir, dataset_name, task_type, metric, limit,
                         output_dir, ensemble_size, ensemble_nbest, seed,
                         shared_mode, max_iterations, precision):
    logger = logging.get_logger(__name__)

    if limit <= 0 and (max_iterations is None or max_iterations <= 0):
        logger.warning("Not starting ensemble builder because it's not worth "
                       "it.")
        # It makes no sense to start building ensembles_statistics
        return
    ensemble_script = 'python -m autosklearn.ensemble_selection_script'
    runsolver_exec = 'runsolver'
    delay = 5

    task_type = TASK_TYPES_TO_STRING[task_type]
    metric = METRIC_TO_STRING[metric]

    call = [ensemble_script,
         '--auto-sklearn-tmp-directory', tmp_dir,
         '--dataset_name', dataset_name,
         '--task', task_type,
         '--metric', metric,
         '--limit', str(limit - 5),
         '--output-directory', output_dir,
         '--ensemble-size', str(ensemble_size),
         '--ensemble-nbest', str(ensemble_nbest),
         '--auto-sklearn-seed', str(seed),
         '--max-iterations', str(max_iterations),
         '--precision', str(precision)]
    if shared_mode:
        call.append('--shared-mode')

    call = ' '.join(call)

    # Runsolver does strange things if the time limit is negative. Set it to
    # be at least one (0 means infinity)
    if limit <= 0:
        limit = 0
    else:
        limit = max(1, limit)

    # Now add runsolver command
    # runsolver_cmd = "%s --watcher-data /dev/null -W %d" % \
    #                (runsolver_exec, limit)
    runsolver_cmd = '%s --watcher-data /dev/null -W %d -d %d' % \
                    (runsolver_exec, limit, delay)
    call = runsolver_cmd + ' ' + call

    proc = submit_call(call, seed, logger, log_dir=tmp_dir)
    return proc
