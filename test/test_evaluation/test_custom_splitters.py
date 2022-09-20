import numpy as np

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
)
from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit

import pytest


@pytest.mark.parametrize(
    "task, X, y",
    [
        (
            BINARY_CLASSIFICATION,
            np.asarray(10000 * [[1, 1, 1, 1, 1]]),
            np.asarray(9999 * [0] + 1 * [1]),
        ),
        (
            MULTICLASS_CLASSIFICATION,
            np.asarray(10000 * [[1, 1, 1, 1, 1]]),
            np.asarray(4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4]),
        ),
        (
            MULTILABEL_CLASSIFICATION,
            np.asarray(10000 * [[1, 1, 1, 1, 1]]),
            np.asarray(
                4999 * [[0, 1, 1]]
                + 4999 * [[1, 1, 0]]
                + 1 * [[1, 0, 1]]
                + 1 * [[0, 0, 0]]
            ),
        ),
    ],
)
@pytest.mark.parametrize("train_size", [100, 0.5, 200, 0.75])
def test_custom_stratified_shuffle_split_returns_unique_labels_and_maintains_size(
    task, X, y, train_size
):
    splitter = CustomStratifiedShuffleSplit(train_size=train_size, random_state=1)
    left_idxs, _ = next(splitter.split(X=X, y=y))
    y_sampled = y[left_idxs]
    X_sampled = X[left_idxs]

    # Assert the train_size param is respected
    if isinstance(train_size, float):
        n_samples = int(train_size * len(y))
    else:
        n_samples = train_size

    assert len(y_sampled) == n_samples
    assert len(X_sampled) == n_samples

    # Assert all the unique labels are present in the training set
    assert all(
        label in np.unique(y_sampled) for label in np.unique(y)
    ), f"{task} failed, {np.unique(y)} != {np.unique(y_sampled)}"
