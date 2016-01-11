import unittest

from autosklearn.pipeline.components.classification.lda import LDA
from autosklearn.pipeline.util import _test_classifier

import sklearn.metrics


class LDAComponentTest(unittest.TestCase):
    def test_default_configuration_iris(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(LDA)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_digits(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(classifier=LDA, dataset='digits')
            self.assertAlmostEqual(0.88585306618093507,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iris_binary(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(LDA, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iris_multilabel(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(LDA, make_multilabel=True)
            self.assertAlmostEqual(0.66,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))
