import unittest

from ParamSklearn.components.regression.libsvm_svr import LibSVM_SVR
from ParamSklearn.util import _test_regressor


import sklearn.metrics


class SupportVectorComponentTest(unittest.TestCase):

    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(LibSVM_SVR)
            self.assertAlmostEqual(0.12849591861430087,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = _test_regressor(LibSVM_SVR,
                                                   sparse=True)
            self.assertAlmostEqual(0.0098877566961463881,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
