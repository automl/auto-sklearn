import operator

import numpy as np


def pareto_front(values: np.ndarray, *, is_loss: bool = True) -> np.ndarray:
    """Calculate the pareto front

    source from: https://stackoverflow.com/a/40239615

    Note
    ----
    Works on the assumption that every value is either something to minimize or
    something to maximize, based on ``is_loss``.

    Parameters
    ----------
    values: np.ndarray [n_models, n_metrics]
        The value for each of the metrics

    is_loss: bool = True
        Whether the metrics are a loss or a score

    Returns
    -------
    np.ndarray
        A boolean mask where true indicates if the model on the pareto front
    """
    op = operator.lt if is_loss else operator.gt

    is_efficient = np.ones(values.shape[0], dtype=bool)
    for i, c in enumerate(values):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(op(values[is_efficient], c), axis=1)

            # And keep self
            is_efficient[i] = True

    return is_efficient
