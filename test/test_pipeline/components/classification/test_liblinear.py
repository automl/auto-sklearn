import unittest

import numpy as np
import sklearn.metrics
import sklearn.svm

from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba


class LibLinearComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_classifier(LibLinear_SVC)
            self.assertTrue(all(targets == predictions))

    def test_default_configuration_sparse(self):
        for i in range(2):
            predictions, targets = _test_classifier(LibLinear_SVC,
                                                    sparse=True)
            self.assertEquals(0.56, sklearn.metrics.accuracy_score(
                targets, predictions))

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(LibLinear_SVC,
                                                    make_binary=True)
            self.assertTrue(all(targets == predictions))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = _test_classifier(LibLinear_SVC,
                                                    make_multilabel=True)
            self.assertAlmostEquals(0.84479797979797977, sklearn.metrics.average_precision_score(
                targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.svm.LinearSVC()

        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)