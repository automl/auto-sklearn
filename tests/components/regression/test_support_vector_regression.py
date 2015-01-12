import unittest

from AutoSklearn.components.regression.support_vector_regression import SupportVectorRegression
from AutoSklearn.util import _test_regressor


import sklearn.metrics


class SupportVectorComponentTest(unittest.TestCase):

    def test_default_configuration(self):

        for i in range(10):
            predictions, targets = _test_regressor(SupportVectorRegression,
                                                   dataset='boston')
            print predictions
            print targets
            self.assertAlmostEqual(-0.070779979927571235,
                sklearn.metrics.r2_score(y_true=targets, y_pred=predictions))
