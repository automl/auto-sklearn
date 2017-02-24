import unittest

from autosklearn.pipeline.components.classification.multinomial_nb import \
    MultinomialNB
from autosklearn.pipeline.util import _test_classifier, _test_classifier_iterative_fit, \
    get_dataset, _test_classifier_predict_proba

import numpy as np
import sklearn.metrics
import sklearn.naive_bayes


class MultinomialNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(MultinomialNB)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_iterative_fit(MultinomialNB)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_negative_values(self):
        # Custon preprocessing test to check if clipping to zero works
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        original_X_train = X_train.copy()
        ss = sklearn.preprocessing.StandardScaler()
        X_train = ss.fit_transform(X_train)
        configuration_space = MultinomialNB.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()

        cls = MultinomialNB(random_state=1, **{hp_name: default[hp_name]
                                               for hp_name in default
                                               if default[hp_name] is not None})

        cls = cls.fit(X_train, Y_train)
        prediction = cls.predict(X_test)
        self.assertAlmostEqual(np.nanmean(prediction == Y_test),
                               0.88888888888888884)

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(MultinomialNB, make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_default_configuration_multilabel(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier(classifier=MultinomialNB,
                                 dataset='digits',
                                 make_multilabel=True)
            self.assertAlmostEqual(0.81239938943608647,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_default_configuration_multilabel_predict_proba(self):
        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(classifier=MultinomialNB,
                                               make_multilabel=True)
            self.assertEqual(predictions.shape, ((50, 3)))
            self.assertAlmostEqual(0.76548981051208942,
                                   sklearn.metrics.average_precision_score(
                                       targets, predictions))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.naive_bayes.MultinomialNB()
        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)