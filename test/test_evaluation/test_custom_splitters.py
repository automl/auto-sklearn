import pytest

import numpy as np

from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit
from autosklearn.constants import (
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION
)

@pytest.mark.parametrize("task, X, y", [
    (
        BINARY_CLASSIFICATION,
        np.asarray(10000 * [[1,1,1,1,1]]),
        np.asarray( 9999 * [0] + 1 * [1])
    ),
    (
        MULTICLASS_CLASSIFICATION,
        np.asarray(10000 * [[1,1,1,1,1]]),
        np.asarray(4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4])),
    (
        MULTILABEL_CLASSIFICATION,
        np.asarray(10000 * [[1,1,1,1,1]]),
        np.asarray(4999 * [[0, 1, 1]] + 4999 * [[1, 1, 0]] + 1 * [[1, 0, 1]] + 1 * [[0, 0, 0]])
    )
])
def test_custom_stratified_shuffle_split_returns_unique_labels(task, X, y):
    splitter = CustomStratifiedShuffleSplit(
        train_size=0.05,
        random_state=1
    )
    left_idxs, _ = next(splitter.split(X=X, y=y))
    y_sampled = y[left_idxs]

    assert all(label in np.unique(y_sampled) for label in np.unique(y)), \
        f"{task} failed, {np.unique(y)} != {np.unique(y_sampled)}"
