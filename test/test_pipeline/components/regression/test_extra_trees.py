import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.8552491508927406
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = 0.8539080454228811
    res["default_boston_sparse"] = 0.4177781339904245
    res["default_boston_iterative_sparse"] = 0.4188803940560917
    res["default_diabetes"] = 0.39073511653135706
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = 0.3900924379086915
    res["default_diabetes_sparse"] = 0.24547114408164694
    res["default_diabetes_iterative_sparse"] = 0.25129575820882655

    sk_mod = sklearn.ensemble.ExtraTreesRegressor
    module = ExtraTreesRegressor
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': module.get_max_iter(),
    }
