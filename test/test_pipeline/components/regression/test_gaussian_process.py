import sklearn.gaussian_process

from autosklearn.pipeline.components.regression.gaussian_process import \
    GaussianProcess

from .test_base import BaseRegressionComponentTest


class GaussianProcessComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston_le_ge"] = [0.6, 0.4]
    res["default_boston_places"] = 1
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = None
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = -7.4131230585194885e-06
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = None
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.gaussian_process.GaussianProcessRegressor

    module = GaussianProcess

    """
    # Leave this here for future reference
    # My machine: 0.574913739659292
    # travis-ci: 0.49562471963524557
    self.assertLessEqual(
        sklearn.metrics.r2_score(y_true=targets, y_pred=predictions),
        0.6)
    self.assertGreaterEqual(
        sklearn.metrics.r2_score(y_true=targets, y_pred=predictions),
        0.4)
    """
