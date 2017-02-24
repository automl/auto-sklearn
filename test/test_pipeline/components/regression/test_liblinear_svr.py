import unittest

from autosklearn.pipeline.components.regression.liblinear_svr import \
    LibLinear_SVR
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics


class SupportVectorComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_regressor(LibLinear_SVR,
                                                   dataset='boston')
            # Lenient test because of travis-ci which gets quite different
            # results here!
            self.assertAlmostEqual(0.68,
                                   sklearn.metrics.r2_score(y_true=targets,
                                                            y_pred=predictions),
                                   places=2)
