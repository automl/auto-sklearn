from typing import Optional, Dict

import unittest

from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_predict_proba, _test_classifier_iterative_fit
from autosklearn.pipeline.constants import SPARSE
from autosklearn.pipeline.components.classification.gradient_boosting import (
    GradientBoostingClassifier
)

import sklearn.metrics
import numpy as np


class BaseClassificationComponentTest(unittest.TestCase):
    # Magic command to not run tests on base class
    __test__ = False

    res = None

    module = None
    sk_module = None
    # Hyperparameter which is increased by iterative_fit
    step_hyperparameter = None

    def test_default_iris(self):

        if self.__class__ == BaseClassificationComponentTest:
            return

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

    def test_get_max_iter(self):
        if self.__class__ == BaseClassificationComponentTest:
            return

        if not hasattr(self.module, 'iterative_fit'):
            return

        self.module.get_max_iter()

    def test_default_iris_iterative_fit(self):

        if self.__class__ == BaseClassificationComponentTest:
            return

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
                    self.res.get("iris_iterative_n_iter", self.step_hyperparameter['value'])
                )

    def test_default_iris_predict_proba(self):

        if self.__class__ == BaseClassificationComponentTest:
            return

        for i in range(2):
            predictions, targets = \
                _test_classifier_predict_proba(dataset="iris",
                                               classifier=self.module)
            self.assertAlmostEqual(self.res["default_iris_proba"],
                                   sklearn.metrics.log_loss(targets, predictions),
                                   places=self.res.get(
                                           "default_iris_proba_places", 7))

    def test_default_iris_sparse(self):

        if self.__class__ == BaseClassificationComponentTest:
            return

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

        if self.__class__ == BaseClassificationComponentTest:
            return

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

        if self.__class__ == BaseClassificationComponentTest:
            return

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

        if self.__class__ == BaseClassificationComponentTest:
            return

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
                    self.res.get("digits_iterative_n_iter", self.step_hyperparameter['value'])
                )

    def test_default_digits_multilabel(self):

        if self.__class__ == BaseClassificationComponentTest:
            return

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

        if self.__class__ == BaseClassificationComponentTest:
            return

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

        if self.__class__ == BaseClassificationComponentTest:
            return

        if not self.module.get_properties()["handles_multiclass"]:
            return
        elif self.sk_module is not None:
            cls = self.sk_module
            X = np.random.random((10, 10))
            y = np.random.randint(0, 1, size=(10, 10))
            self.assertRaisesRegex(
                ValueError,
                'bad input shape \\(10, 10\\)',
                cls.fit,
                X,
                y
            )
        else:
            return

    def test_module_idempotent(self):
        """ Fitting twice with the same config gives the same model params. """
        if self.__class__ == BaseClassificationComponentTest:
            return

        classifier_cls = self.module

        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
        ])
        y = np.array([
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
        ])

        # There are certain errors we ignore so we wrap this in a function
        def fitted_params(model) -> Optional[Dict]:
            """
            Returns the params if fitted successfully, else None if an
            acceptable error occurs
            """
            # We are okay with Numerical in Quadractic disciminant analysis
            def is_QDA_error(err):
                return "Numerical problems in QDA" in err.args[0]

            # We are okay if the BaseClassifier in AdaBoostClassifier is worse
            # than random so no ensemble can be fit
            def is_AdaBoostClassifier_error(err):
                return ("BaseClassifier in AdaBoostClassifier ensemble is worse"
                        + " than random, ensemble can not be fit." in err.args[0])

            def is_unset_param_raw_predictions_val_error(err):
                return ("local variable 'raw_predictions_val' referenced before"
                        + " assignment" in err.args[0])

            try:
                model.fit(X.copy(), y.copy())
            except ValueError as e:
                if is_AdaBoostClassifier_error(e) or is_QDA_error(e):
                    return None
            except UnboundLocalError as e:
                if is_unset_param_raw_predictions_val_error(e):
                    return None

            return model.estimator.get_params()

        # We ignore certain keys when comparing
        param_keys_ignored = [
            'random_state', 'base_estimator', *self.res.get('ignore_hps', [])
        ]

        # We use the default config + sampled ones
        configuration_space = classifier_cls.get_hyperparameter_search_space()

        default = configuration_space.get_default_configuration()
        sampled = [configuration_space.sample_configuration() for _ in range(5)]

        for config in [default] + sampled:
            model_args = {
                'random_state': np.random.RandomState(1),
                ** {
                    hp_name: config[hp_name]
                    for hp_name in config
                    if config[hp_name] is not None
                }
            }
            classifier = classifier_cls(**model_args)

            # Get the parameters on the first and second fit with config params
            params_first = fitted_params(classifier)
            params_second = fitted_params(classifier)

            # An acceptable error occured, skip to next sample
            if params_first is None or params_second is None:
                continue

            # Remove keys we don't wish to include in the comparison
            for params in [params_first, params_second]:
                for key in param_keys_ignored:
                    if key in params:
                        del params[key]

            if params_first != params_second:
                print(model_args)

            # They should be equal
            self.assertEqual(params_first, params_second)

    @unittest.skip("Issue 1209")
    def test_gradient_boosting_module_idempotent_max_iter(self):
        """
        Succesive calls to `fit` GradientBoostingClassifier actually call
        iterativ fit. This means the estimator produced after two successive
        calls does not result in the same output estimator.
        """
        if self.module != GradientBoostingClassifier:
            pass

        # Gotten from failing runs of test_module_idempotent
        model_args = {
            'random_state': np.random.RandomState(1),
            'early_stop': 'valid',
            'l2_regularization': 2.125626112922482e-06,
            'learning_rate': 0.09554502382606479,
            'loss': 'auto',
            'max_bins': 255,
            'max_depth': 'None',
            'max_leaf_nodes': 15,
            'min_samples_leaf': 3,
            'scoring': 'loss',
            'tol': 1e-07,
            'n_iter_no_change': 11,
            'validation_fraction': 0.2751790689986756
        }

        X = np.array([
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 0], [0, 1], [1, 0], [1, 1],
        ])
        y = np.array([
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
        ])

        classifier = self.module(**model_args)

        # Get the parameters on the first and second fit with config params
        params_first = classifier.fit(X, y).estimator.get_params()
        params_second = classifier.fit(X, y).estimator.get_params()

        # Remove keys we don't wish to include in the comparison
        # We ignore certain keys when comparing
        assert params_first['max_iter'] == params_second['max_iter']
