import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.8547046708967643
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = 0.8535511298779792
    res["default_boston_sparse"] = 0.419556142451604
    res["default_boston_iterative_sparse"] = 0.4164715852754325
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
