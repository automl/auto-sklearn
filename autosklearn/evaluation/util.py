import queue

from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, CLASSIFICATION_METRICS, \
    REGRESSION_METRICS


__all__ = [
    'calculate_score',
    'get_last_result'
]


def calculate_score(solution, prediction, task_type, metric, all_scoring_functions=False):

    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            for metric_ in REGRESSION_METRICS:
                func = REGRESSION_METRICS[metric_]
                score[func.name] = func(solution, cprediction)

        else:
            for metric_ in CLASSIFICATION_METRICS:
                func = CLASSIFICATION_METRICS[metric_]

                # TODO maybe annotate metrics to define which cases they can
                # handle?

                try:
                    score[func.name] = func(solution, prediction)
                except ValueError as e:
                    if e.args[0] == 'multiclass format is not supported':
                        continue
                    elif e.args[0] == 'Sample-based precision, recall, ' \
                                      'fscore is not meaningful outside ' \
                                      'multilabel classification. See the ' \
                                      'accuracy_score instead.':
                        continue
                    else:
                        raise e

    else:
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            score = metric(solution, cprediction)
        else:
            score = metric(solution, prediction)

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
