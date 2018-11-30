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
    # Hyperparameter which is increased by iterative_fit
    step_hyperparameter = None

    # Magic command to not run tests on base class
    __test__ = False

    def test_default_iris(self):
        for i in range(2):
            predictions, targets, n_calls = \
                _test_classifier(dataset="iris",
                                 classifier=self.module)
            self.assertAlmostEqual(self.res["default_iris"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_iris_places", 7))

            if self.res.get("iris_n_calls"):
                self.assertEqual(self.res["iris_n_calls"], n_calls)

    def test_default_iris_iterative_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets, classifier = \
                _test_classifier_iterative_fit(dataset="iris",
                                               classifier=self.module)
            self.assertAlmostEqual(self.res["default_iris_iterative"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_iris_iterative_places", 7))

            if self.step_hyperparameter is not None:
                self.assertEqual(
                    getattr(classifier.estimator, self.step_hyperparameter['name']),
                    self.step_hyperparameter['value']
                )


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
            predictions, targets, _ = \
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
            predictions, targets, _ = \
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
            predictions, targets, n_calls = \
                _test_classifier(dataset="digits",
                                 classifier=self.module)
            self.assertAlmostEqual(self.res["default_digits"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_digits_places", 7))

            if self.res.get("digits_n_calls"):
                self.assertEqual(self.res["digits_n_calls"], n_calls)

    def test_default_digits_iterative_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets, classifier = \
                _test_classifier_iterative_fit(dataset="digits",
                                               classifier=self.module)
            self.assertAlmostEqual(self.res["default_digits_iterative"],
                                   sklearn.metrics.accuracy_score(targets,
                                                                  predictions),
                                   places=self.res.get(
                                           "default_digits_iterative_places", 7))

            if self.step_hyperparameter is not None:
                self.assertEqual(
                    getattr(classifier.estimator, self.step_hyperparameter['name']),
                    self.step_hyperparameter['value']
                )

    def test_default_digits_multilabel(self):
        if not self.module.get_properties()["handles_multilabel"]:
            return

        for i in range(2):
            predictions, targets, _ = \
                _test_classifier(classifier=self.module,
                                 dataset='digits',
                                 make_multilabel=True)
            self.assertAlmostEqual(self.res["default_digits_multilabel"],
                                   sklearn.metrics.precision_score(
                                       targets, predictions, average='macro'),
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
                                   sklearn.metrics.roc_auc_score(
                                       targets, predictions, average='macro'),
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

    def test_module_idempotent(self):
        def check_classifier(cls):
            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            y = np.array([0, 1, 1, 0])
            params = []

            for i in range(2):
                try:
                    classifier.fit(X, y)
                except ValueError as e:
                    if (
                        isinstance(e.args[0], str)
                    ) and (
                        "Numerical problems in QDA" in e.args[0]
                    ):
                        continue
                    elif (
                        "BaseClassifier in AdaBoostClassifier ensemble is "
                        "worse than random, ensemble can not be fit." in e.args[0]
                    ):
                        continue
                    else:
                        raise e

                p = classifier.estimator.get_params()
                if 'random_state' in p:
                    del p['random_state']
                if 'base_estimator' in p:
                    del p['base_estimator']
                for ignore_hp in self.res.get('ignore_hps', []):
                    del p[ignore_hp]
                params.append(p)

                if i > 0:
                    self.assertEqual(
                        params[-1],
                        params[0],
                    )

        classifier = self.module
        configuration_space = classifier.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        classifier = classifier(random_state=np.random.RandomState(1),
                                **{hp_name: default[hp_name] for hp_name in
                                   default if default[hp_name] is not None})
        check_classifier(classifier)

        for i in range(10):
            classifier = self.module
            config = configuration_space.sample_configuration()
            classifier = classifier(random_state=np.random.RandomState(1),
                                    **{hp_name: config[hp_name] for hp_name in
                                       config if config[hp_name] is not None})
            check_classifier(classifier)
