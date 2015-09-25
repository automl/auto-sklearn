import os

import lockfile
import numpy as np

from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, a_metric, r2_metric, \
    normalize_array, bac_metric, auc_metric, f1_metric, pac_metric, \
    acc_metric, regression_metrics, classification_metrics


__all__ = [
    'calculate_score',
    'get_new_run_num'
]


def calculate_score(solution, prediction, task_type, metric, num_classes,
                    all_scoring_functions=False):
    if task_type == MULTICLASS_CLASSIFICATION:
        solution_binary = np.zeros((prediction.shape[0], num_classes))
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
            score['a_metric'] = a_metric(solution, cprediction,
                                         task=task_type)
            score['r2_metric'] = r2_metric(solution, cprediction,
                                           task=task_type)
        else:
            csolution, cprediction = normalize_array(solution, prediction)
            score['bac_metric'] = bac_metric(csolution, cprediction,
                                             task=task_type)
            score['auc_metric'] = auc_metric(csolution, cprediction,
                                             task=task_type)
            score['f1_metric'] = f1_metric(csolution, cprediction,
                                           task=task_type)
            score['pac_metric'] = pac_metric(csolution, cprediction,
                                             task=task_type)
            score['acc_metric'] = acc_metric(csolution, cprediction,
                                             task=task_type)

    else:
        if task_type in REGRESSION_TASKS:
            scoring_func = getattr(regression_metrics, metric)
            cprediction = sanitize_array(prediction)
            score = scoring_func(solution, cprediction, task=task_type)
        else:
            scoring_func = getattr(classification_metrics, metric)
            csolution, cprediction = normalize_array(solution, prediction)
            score = scoring_func(csolution, cprediction, task=task_type)
    return score


def get_new_run_num():
    seed = os.environ.get('AUTOSKLEARN_SEED')
    counter_file = 'num_run'
    if seed is not None:
        counter_file = counter_file + ('_%s' % seed)
    counter_file = os.path.join(os.getcwd(), counter_file)
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