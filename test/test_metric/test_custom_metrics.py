import random

import numpy as np

import pytest

from sklearn.utils.multiclass import type_of_target

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
)
from autosklearn.metrics import (
    CLASSIFICATION_METRICS,
    CUSTOM_METRICS,
    calculate_score,
    clean_predictions,
)


@pytest.fixture
def custom_metric_y_true_pred(request):
    if request.param == 'binary_same_dim':
        y_true = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0])
        return y_true, y_pred
    elif request.param == 'binary_diff_dim':
        y_true = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.eye(2)[y_pred]
        return y_true, y_pred
    elif request.param == 'binary_randomized':
        y_true = np.random.choice([0, 1], size=100)
        y_pred = np.random.choice([0, 1], size=100)
        return y_true, y_pred
    elif request.param == 'multiclass_same_dim':
        y_true = np.array([0, 1, 1, 0, 2, 0, 0, 4, 1, 3, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 2, 1, 0, 0, 1, 3, 1, 4])
        return y_true, y_pred
    elif request.param == 'multiclass_diff_dim':
        y_true = np.array([0, 1, 1, 0, 2, 0, 0, 4, 1, 3, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 2, 1, 0, 0, 1, 3, 1, 4])
        y_pred = np.eye(5)[y_pred]
        return y_true, y_pred
    elif request.param == 'multiclass_randomized':
        y_true = np.random.choice([0, 1, 2], size=100)
        y_pred = np.random.choice([0, 1, 2], size=100)
        return y_true, y_pred
    elif request.param == 'multilabel':
        y_true = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0], [0, 1, 1],
                           [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1]
                           ])
        y_pred = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 1],
                           [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1]
                           ])
        return y_true, y_pred
    elif request.param == 'multilabel_randomized':
        a_list = [[1, 0, 1], [0, 1, 1], [0, 0, 0]]
        y_true = np.array(list(map(lambda x: random.choice(a_list), range(100))))
        y_pred = np.array(list(map(lambda x: random.choice(a_list), range(100))))
        return y_true, y_pred
    else:
        raise ValueError(f"Unsupported fixture request {request.param}")


@pytest.mark.parametrize("custom_metric_y_true_pred",
                         ['binary_same_dim', 'binary_randomized', 'multiclass_same_dim',
                          'multiclass_randomized', 'multilabel', 'multilabel_randomized',
                          # Do a couple of randomizations
                          'binary_randomized', 'multiclass_randomized', 'multilabel_randomized',
                          'binary_randomized', 'multiclass_randomized', 'multilabel_randomized',
                          ],
                         indirect=True)
@pytest.mark.parametrize("metric", ['accuracy', 'balanced_accuracy', 'log_loss', 'roc_auc'])
def test_custom_metric_score(metric, custom_metric_y_true_pred):

    sklearn_score_func = CLASSIFICATION_METRICS[metric]
    custom_implementation = CUSTOM_METRICS[metric]
    y_true, y_pred = custom_metric_y_true_pred

    if type_of_target(y_true) == 'multilabel-indicator' and metric == 'balanced_accuracy':
        pytest.skip('Balanced accuracy not supported for multilabel')
    if type_of_target(y_true) == 'multiclass' and metric == 'roc_auc':
        pytest.skip('ROC AUC not supported for multiclass')

    # In case of log loss, we need probabilities
    if metric == 'log_loss':
        num_classes = len(np.unique(y_true))
        if len(y_pred.shape) == 1 or y_pred.shape[1] < num_classes:
            y_pred = np.eye(num_classes)[y_pred]

    expected = sklearn_score_func(y_true, y_pred)
    score = custom_implementation(y_true, y_pred)
    assert expected == pytest.approx(score)


@pytest.mark.parametrize("custom_metric_y_true_pred",
                         ['binary_same_dim', 'binary_randomized', 'multiclass_same_dim',
                          'multiclass_randomized', 'multilabel', 'multilabel_randomized',
                          # Add different dimension scenarios that clean-prediction solves
                          'binary_diff_dim', 'multiclass_diff_dim',
                          # Do a couple of randomizations
                          'binary_randomized', 'multiclass_randomized', 'multilabel_randomized',
                          'binary_randomized', 'multiclass_randomized', 'multilabel_randomized',
                          ],
                         indirect=True)
@pytest.mark.parametrize("metric", ['accuracy', 'balanced_accuracy', 'log_loss', 'roc_auc'])
def test_custom_metric_in_scorer(metric, custom_metric_y_true_pred):
    sklearn_score_func = CLASSIFICATION_METRICS[metric]
    custom_implementation = CUSTOM_METRICS[metric]
    y_true, y_pred = custom_metric_y_true_pred
    if metric == 'log_loss':
        # In case of log loss, we need probabilities
        num_classes = len(np.unique(y_true))
        if len(y_pred.shape) == 1 or y_pred.shape[1] < num_classes:
            y_pred = np.eye(num_classes)[y_pred]
    task_type = type_of_target(y_true)
    task_mapping = {'multilabel-indicator': MULTILABEL_CLASSIFICATION,
                    'multiclass': MULTICLASS_CLASSIFICATION,
                    'binary': BINARY_CLASSIFICATION}

    if type_of_target(y_true) == 'multilabel-indicator' and metric == 'balanced_accuracy':
        pytest.skip('Balanced accuracy not supported for multilabel')
    if type_of_target(y_true) == 'multiclass' and metric == 'roc_auc':
        pytest.skip('ROC AUC not supported for multiclass')

    sklearn_score = calculate_score(
        solution=y_true,
        prediction=y_pred,
        task_type=task_mapping[task_type],
        metric=sklearn_score_func,
    )
    predictions = clean_predictions(custom_implementation, y_pred, task_type)
    fast_score = calculate_score(
        solution=y_true,
        prediction=predictions,
        task_type=task_mapping[task_type],
        metric=custom_implementation,
    )
    assert sklearn_score == fast_score
