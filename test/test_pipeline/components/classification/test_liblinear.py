import unittest

import sklearn.metrics

from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.util import _test_classifier


class LibLinearComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = _test_classifier(LibLinear_SVC)
            self.assertTrue(all(targets == predictions))

    def test_default_configuration_sparse(self):
        for i in range(10):
            predictions, targets = _test_classifier(LibLinear_SVC,
                                                    sparse=True)
            self.assertEquals(0.56, sklearn.metrics.accuracy_score(
                targets, predictions))

    def test_default_configuration_binary(self):
        for i in range(10):
            predictions, targets = _test_classifier(LibLinear_SVC,
                                                    make_binary=True)
            self.assertTrue(all(targets == predictions))

    def test_default_configuration_multilabel(self):
        for i in range(10):
            predictions, targets = _test_classifier(LibLinear_SVC,
                                                    make_multilabel=True)
            self.assertAlmostEquals(0.84479797979797977, sklearn.metrics.average_precision_score(
                targets, predictions))