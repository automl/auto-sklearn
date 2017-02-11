import queue

from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, \
    regression_metrics, classification_metrics, get_all_known_metrics, get_metric


__all__ = [
    'calculate_score',
    'get_last_result'
]


def calculate_score(solution, prediction, task_type, metric, num_classes,
                    all_scoring_functions=False, logger=None):
    metric = get_metric(metric, task_type)

    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            prediction = sanitize_array(prediction)

        for metric_ in get_all_known_metrics(task_type):
            score[metric_.name] = metric_.calculate_score(solution, prediction)

    else:
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            prediction = sanitize_array(prediction)
        score = metric.calculate_score(solution, prediction)

    return score


def get_last_result(queue_):
    stack = []
    while True:
        try:
            rval = queue_.get(timeout=1)
        except queue.Empty:
            break
        stack.append(rval)
    return stack.pop()
