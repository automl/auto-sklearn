import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.8457463981374789
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.8460054319822927
    res["default_boston_sparse"] = 0.4022606040314697
    res["default_boston_iterative_sparse"] = 0.4241336980285072
    res["default_diabetes"] = 0.38268255933451
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.38869684680884398
    res["default_diabetes_sparse"] = 0.2436985361045233
    res["default_diabetes_iterative_sparse"] = 0.25345808769763623

    sk_mod = sklearn.ensemble.ExtraTreesRegressor
    module = ExtraTreesRegressor
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }
