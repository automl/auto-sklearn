import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.validation import check_is_fitted

from autosklearn.evaluation.abstract_evaluator import (
    MyDummyClassifier,
    MyDummyRegressor,
)

import pytest


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_dummy_pipeline(task_type: str) -> None:
    if task_type == "classification":
        estimator_class = MyDummyClassifier
        data_maker = make_classification
    elif task_type == "regression":
        estimator_class = MyDummyRegressor
        data_maker = make_regression
    else:
        pytest.fail(task_type)
        return

    X, y = data_maker(random_state=0)
    estimator = estimator_class(
        feat_type={i: "numerical" for i in range(X.shape[1])}, config=1, random_state=0
    )
    estimator.fit(X, y)
    check_is_fitted(estimator)

    assert np.shape(X)[0] == np.shape(estimator.predict(X))[0]

    # make sure we comply with scikit-learn estimator API
    clone(estimator)
