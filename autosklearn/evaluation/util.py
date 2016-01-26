import os

import lockfile
import numpy as np

from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, \
    regression_metrics, classification_metrics, create_multiclass_solution


__all__ = [
    'calculate_score',
    'get_new_run_num'
]


def calculate_score(solution, prediction, task_type, metric, num_classes,
                    all_scoring_functions=False, logger=None):
    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            for metric_ in REGRESSION_METRICS:
                score[metric_] = regression_metrics.calculate_score(
                    metric_, solution, cprediction)
        else:
            for metric_ in CLASSIFICATION_METRICS:
                score[metric_] = classification_metrics.calculate_score(
                    metric_, solution, prediction, task_type)

    else:
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            score = regression_metrics.calculate_score(
                metric, solution, cprediction)
        else:
            score = classification_metrics.calculate_score(
                metric, solution, prediction, task=task_type)
    return score


def get_new_run_num(lockdir=None):
    seed = os.environ.get('AUTOSKLEARN_SEED')
    counter_file = 'num_run'
    if seed is not None:
        counter_file = counter_file + ('_%s' % seed)
    if lockdir is None:
        lockdir = os.getcwd()
    counter_file = os.path.join(lockdir, counter_file)
    lock = lockfile.LockFile(counter_file)
    with lock:
        if not os.path.exists(counter_file):
            with open(counter_file, 'w') as fh:
                fh.write('0')
            num = 0
        else:
            with open(counter_file, 'r') as fh:
                num = int(fh.read())
            num += 1
            with open(counter_file, 'w') as fh:
                fh.write(str(num).zfill(4))

    return num
