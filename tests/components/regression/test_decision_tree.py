import unittest

from ParamSklearn.components.regression.decision_tree import DecisionTree
from ParamSklearn.util import _test_regressor

import sklearn.metrics


class DecisionTreetComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_regressor(DecisionTree,)
            self.assertAlmostEqual(0.14886750572325669,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = _test_regressor(DecisionTree, sparse=True)
            self.assertAlmostEqual(0.021778487309118133,
                                   sklearn.metrics.r2_score(targets,
                                                            predictions))
