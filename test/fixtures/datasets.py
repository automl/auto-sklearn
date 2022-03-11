from typing import Tuple, Callable

import numpy as np
from pytest import fixture

from autosklearn.pipeline.util import get_dataset

# Technically not true as make_datset can return spmatrix
TrainTestSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# TODO Remove the implementation in autosklearn.pipeline.util and just put here
@fixture
def get_sklearn_dataset() -> Callable[..., TrainTestSet]:
    """

    Parameters
    ----------
    name : str = "iris"
        Name of the dataset to get

    make_sparse : bool = False
        Wehther to make the data sparse

    add_NaNs : bool = False
        Whether to add NaNs to the data

    train_size_maximum : int = 150
        THe maximum size of training data

    make_multilabel : bool = False
        Whether to force the data into being multilabel

    make_binary : bool = False
        Whether to force the data into being binary

    Returns
    -------
    (X_train, Y_train, X_test, Y_Test)
    """
    def _get(
        name: str = "iris",
        make_sparse: bool = False,
        add_NaNs: bool = False,
        train_size_maximum: int = 150,
        make_multilabel: bool = False,
        make_binary: bool = False,
    ) -> TrainTestSet:
        return get_dataset(
            dataset=name,
            make_sparse=make_sparse,
            add_NaNs=add_NaNs,
            train_size_maximum=train_size_maximum,
            make_multilabel=make_multilabel,
            make_binary=make_binary,
        )

    return _get
