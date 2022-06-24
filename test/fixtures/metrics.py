from typing import Any

import numpy as np

from autosklearn.metrics import accuracy, make_scorer


def _accuracy_requiring_X_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_data: Any,
) -> float:
    """Dummy metric that needs X Data"""
    if X_data is None:
        raise ValueError()
    return accuracy(y_true, y_pred)


acc_with_X_data = make_scorer(
    name="acc_with_X_data",
    score_func=_accuracy_requiring_X_data,
    needs_X=True,
    optimum=1,
    worst_possible_result=0,
    greater_is_better=True,
)
