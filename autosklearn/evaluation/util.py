import os

import lockfile
import numpy as np

from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, \
    normalize_array, regression_metrics, classification_metrics


__all__ = [
    'calculate_score',
    'get_new_run_num'
]


def calculate_score(solution, prediction, task_type, metric, num_classes,
                    all_scoring_functions=False, logger=None):
    if task_type == MULTICLASS_CLASSIFICATION:
        # This used to crash on travis-ci; special treatment to find out why
        # it crashed!
        try:
            solution_binary = np.zeros((prediction.shape[0], num_classes))
        except IndexError as e:
            if logger is not None:
                logger.error("Prediction shape: %s, solution "
                             "shape %s", prediction.shape, solution.shape)
                raise e

        #indices = np.ones(solution_binary.shape[0], dtype=int) * solution
        #solution_binary[:, indices] = 1.0

        for i in range(solution_binary.shape[0]):
            label = solution[i]
            solution_binary[i, label] = 1
        solution = solution_binary

    elif task_type in [BINARY_CLASSIFICATION, REGRESSION]:
        if len(solution.shape) == 1:
            solution = solution.reshape((-1, 1))

    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    if solution.shape != prediction.shape:
        raise ValueError('Solution shape %s != prediction shape %s' %
                         (solution.shape, prediction.shape))

    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
            cprediction = sanitize_array(prediction)
            for metric_ in REGRESSION_METRICS:
                score[metric_] = regression_metrics.calculate_score(metric_,
                                                                    solution,
                                                                    cprediction)
        else:
            csolution, cprediction = normalize_array(solution, prediction)
            for metric_ in CLASSIFICATION_METRICS:
                score[metric_] = classification_metrics.calculate_score(
                    metric_, csolution, cprediction, task_type)

    else:
        if task_type in REGRESSION_TASKS:
            cprediction = sanitize_array(prediction)
            score = regression_metrics.calculate_score(metric,
                                                       solution,
                                                       cprediction)
        else:
            csolution, cprediction = normalize_array(solution, prediction)
            score = classification_metrics.calculate_score(metric,
                                                           csolution,
                                                           cprediction,
                                                           task=task_type)
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
