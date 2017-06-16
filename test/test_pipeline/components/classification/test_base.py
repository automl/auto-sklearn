import unittest

from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_predict_proba, _test_classifier_iterative_fit
from autosklearn.pipeline.constants import *

import sklearn.metrics
import numpy as np


class BaseClassificationComponentTest(unittest.TestCase):

    res = None

    module = None
    sk_module = None

    # Magic command to not run tests on base class
    __test__ = False

    def test_default_iris(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(dataset="iris",
                                 classifier=self.module)
            self.assertAlmostEqual(self.res["default_iris"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_iris_places", 7))

    def test_default_iris_iterative_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(dataset="iris",
                                               classifier=self.module)
            self.assertAlmostEqual(self.res["default_iris_iterative"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_iris_iterative_places", 7))

    def test_default_iris_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(dataset="iris",
                                               classifier=self.module)
            self.assertAlmostEqual(self.res["default_iris_proba"],
                                   sklearn.metrics.log_loss(targets, predictions),
                                   places=self.res.get(
                                           "default_iris_proba_places", 7))

    def test_default_iris_sparse(self):
        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets = \
                _test_classifier(dataset="iris",
                                 classifier=self.module,
                                 sparse=True)
            self.assertAlmostEqual(self.res["default_iris_sparse"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_iris_sparse_places", 7))

    def test_default_digits_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=self.module,
                                 dataset='digits', sparse=False,
                                 make_binary=True)
            self.assertAlmostEqual(self.res["default_digits_binary"],
                                   sklearn.metrics.accuracy_score(
                                       targets, predictions),
                                   places=self.res.get(
                                           "default_digits_binary_places", 7))

    def test_default_digits(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(dataset="digits",
                                 classifier=self.module)
            self.assertAlmostEqual(self.res["default_digits"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_digits_places", 7))

    def test_default_digits_iterative_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(dataset="digits",
                                               classifier=self.module)
            self.assertAlmostEqual(self.res["default_digits_iterative"],
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets),
                                   places=self.res.get(
                                           "default_digits_iterative_places", 7))

    def test_default_digits_multilabel(self):
        if not self.module.get_properties()["handles_multilabel"]:
            return

        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=self.module,
                                 dataset='digits',
                                 make_multilabel=True)
            self.assertAlmostEqual(self.res["default_digits_multilabel"],
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions),
                                   places=self.res.get(
                                           "default_digits_multilabel_places", 7))

    def test_default_digits_multilabel_predict_proba(self):
        if not self.module.get_properties()["handles_multilabel"]:
            return

        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(classifier=self.module,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(self.res["default_digits_multilabel_proba"],
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions),
                                   places=self.res.get(
                                           "default_digits_multilabel_proba_places", 7))

    def test_target_algorithm_multioutput_multiclass_support(self):
        if not self.module.get_properties()["handles_multiclass"]:
            return
        elif self.sk_module is not None:
            cls = self.sk_module
            X = np.random.random((10, 10))
            y = np.random.randint(0, 1, size=(10, 10))
            self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                    cls.fit, X, y)
        else:
            return 
