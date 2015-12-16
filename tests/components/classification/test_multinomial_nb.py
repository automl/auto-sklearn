import unittest

from ParamSklearn.components.classification.multinomial_nb import \
    MultinomialNB
from ParamSklearn.util import _test_classifier, _test_classifier_iterative_fit, \
    get_dataset

import numpy as np
import sklearn.metrics


class MultinomialNBComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(10):
            predictions, targets = \
                _test_classifier(MultinomialNB)
            self.assertAlmostEqual(0.97999999999999998,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  targets))

    def test_default_configuration_iterative_fit(self):
        for i in range(10):
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