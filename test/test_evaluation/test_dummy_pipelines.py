import numpy as np

import pytest

from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.validation import check_is_fitted

from autosklearn.evaluation.abstract_evaluator import MyDummyClassifier, MyDummyRegressor


@pytest.mark.parametrize("task_type", ['classification', 'regression'])
def test_dummy_pipeline(task_type):
    if task_type == 'classification':
        estimator_class = MyDummyClassifier
        data_maker = make_classification
    elif task_type == 'regression':
        estimator_class = MyDummyRegressor
        data_maker = make_regression
    else:
        pytest.fail(task_type)
        return

    estimator = estimator_class(config=1, random_state=0)
    X, y = data_maker(random_state=0)
    estimator.fit(X, y)
    check_is_fitted(estimator)

    assert np.shape(X)[0] == np.shape(estimator.predict(X))[0]

    # make sure we comply with scikit-learn estimator API
    clone(estimator)
