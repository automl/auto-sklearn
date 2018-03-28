import numpy as np

import sklearn.naive_bayes
import sklearn.preprocessing

from autosklearn.pipeline.components.classification.multinomial_nb import \
    MultinomialNB
from autosklearn.pipeline.util import get_dataset

from .test_base import BaseClassificationComponentTest


class MultinomialNBComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.97999999999999998
    res["iris_n_calls"] = 1
    res["default_iris_iterative"] = 0.97999999999999998
    res["default_iris_proba"] = 0.5879188799085624
    res["default_iris_sparse"] = 0.54
    res["default_digits"] = 0.89496053430479661
    res["digits_n_calls"] = 1
    res["default_digits_iterative"] = 0.89496053430479661
    res["default_digits_binary"] = 0.98967820279295693
    res["default_digits_multilabel"] = 0.70484946987667163
    res["default_digits_multilabel_proba"] = 0.80324074074074081

    sk_mod = sklearn.naive_bayes.MultinomialNB
    module = MultinomialNB

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
