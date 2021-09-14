import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.8539264243687228
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = res['default_boston']
    res["default_boston_sparse"] = 0.411211701806908
    res["default_boston_iterative_sparse"] = res['default_boston_sparse']
    res["default_diabetes"] = 0.3885150255877827
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = res['default_diabetes']
    res["default_diabetes_sparse"] = 0.2422804139169642
    res["default_diabetes_iterative_sparse"] = res['default_diabetes_sparse']

    sk_mod = sklearn.ensemble.ExtraTreesRegressor
    module = ExtraTreesRegressor
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': module.get_max_iter(),
    }
