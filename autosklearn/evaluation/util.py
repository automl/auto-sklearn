from autosklearn.constants import *
from autosklearn.metrics import sanitize_array, \
    regression_metrics, classification_metrics


__all__ = [
    'calculate_score',
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
