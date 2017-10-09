import sklearn.svm

from autosklearn.pipeline.components.regression.liblinear_svr import \
    LibLinear_SVR
from .test_base import BaseRegressionComponentTest


class SupportVectorComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.6768297818275556
    res["default_boston_places"] = 2
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = 0.12626519114138912
    res["default_boston_sparse_places"] = 2
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.39152218711865661
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = 0.18704323088631891
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.svm.LinearSVR

    module = LibLinear_SVR
