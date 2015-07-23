import unittest

from ParamSklearn.components.regression.liblinear_svr import \
    LibLinear_SVR
from ParamSklearn.util import _test_regressor

import sklearn.metrics


class SupportVectorComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(LibLinear_SVR,
                                                   dataset='boston')
            self.assertAlmostEqual(0.54372712745256768,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions))
