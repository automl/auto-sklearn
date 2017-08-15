import sklearn.tree

from autosklearn.pipeline.components.regression.decision_tree import \
    DecisionTree
from .test_base import BaseRegressionComponentTest


class DecisionTreeComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.35616796434879905
    res["default_boston_iterative"] = -1
    res["default_boston_sparse"] = 0.18031669797027394
    res["default_boston_iterative_sparse"] = 0.0
    res["default_diabetes"] = 0.1564592449511697
    res["default_diabetes_iterative"] = -1
    res["default_diabetes_sparse"] = -0.020818312539637507
    res["default_diabetes_iterative_sparse"] = 0.0

    sk_mod = sklearn.tree.DecisionTreeRegressor

    module = DecisionTree