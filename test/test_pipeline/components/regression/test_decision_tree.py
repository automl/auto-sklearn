import unittest

from autosklearn.pipeline.components.regression.decision_tree import DecisionTree
from autosklearn.pipeline.util import _test_regressor

import sklearn.metrics


class DecisionTreetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_regressor(DecisionTree,)
            self.assertAlmostEqual(0.1564592449511697,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = _test_regressor(DecisionTree, sparse=True)
            self.assertAlmostEqual(-0.020818312539637507,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
