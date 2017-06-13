import unittest

from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics
import numpy as np


class BaseClassificationComponentTest(unittest.TestCase):

    res_dict = None
    module = None
    sk_module = None

    # Magic command to not run tests on base class
    __test__ = False

    def test_default_configuration_iris(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(self.module)
            self.assertAlmostEqual(self.res["default_on_iris"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_iris_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(self.module)
            self.assertAlmostEqual(self.res["default_on_iris_proba"],
                                   sklearn.metrics.log_loss(targets, predictions))

    def test_default_configuration_iris_sparse(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(self.module, sparse=True)
            self.assertAlmostEqual(self.res["default_on_iris_sparse"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=self.module,
                                 dataset='digits', sparse=True,
                                 make_binary=True)
            self.assertAlmostEqual(self.res["default_binary"],
                                   sklearn.metrics.accuracy_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        if self.sk_module is not None:
            cls = self.sk_module
            X = np.random.random((10, 10))
            y = np.random.randint(0, 1, size=(10, 10))
            self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                    cls.fit, X, y)
        else:
            return 
