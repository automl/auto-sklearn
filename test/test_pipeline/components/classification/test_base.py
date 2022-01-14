from typing import Optional, Dict

import unittest

from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_predict_proba, _test_classifier_iterative_fit
from autosklearn.pipeline.constants import SPARSE

import sklearn.metrics
import numpy as np

from test.test_pipeline.ignored_warnings import ignore_warnings, classifier_warnings


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

        for _ in range(2):
            predictions, targets = _test_classifier_predict_proba(
                dataset="iris", classifier=self.module
            )
            self.assertAlmostEqual(
                self.res["default_iris_proba"],
                sklearn.metrics.log_loss(targets, predictions),
                places=self.res.get("default_iris_proba_places", 7)
            )

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

        for _ in range(2):
            predictions, targets, _ = _test_classifier(
                classifier=self.module, dataset='digits', make_multilabel=True
            )

            score = sklearn.metrics.precision_score(
                targets, predictions, average='macro', zero_division=0
            )
            self.assertAlmostEqual(
                self.res["default_digits_multilabel"], score,
                places=self.res.get("default_digits_multilabel_places", 7)
            )

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
        """ Fitting twice with the same config gives the same model params.

            This is only valid when the random_state passed is an int. If a
            RandomState object is passed then repeated calls to fit will have
            different results. See the section on "Controlling Randomness" in the
            sklearn docs.

            https://scikit-learn.org/0.24/common_pitfalls.html#controlling-randomness
        """
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
                with ignore_warnings(classifier_warnings):
                    model.fit(X.copy(), y.copy())
            except ValueError as e:
                if is_AdaBoostClassifier_error(e) or is_QDA_error(e):
                    return None
            except UnboundLocalError as e:
                if is_unset_param_raw_predictions_val_error(e):
                    return None

            return model.estimator.get_params()

        # We ignore certain keys when comparing
        param_keys_ignored = ['base_estimator']

        # We use the default config + sampled ones
        configuration_space = classifier_cls.get_hyperparameter_search_space()

        default = configuration_space.get_default_configuration()
        sampled = [configuration_space.sample_configuration() for _ in range(2)]

        for seed, config in enumerate([default] + sampled):
            model_args = {"random_state": seed, **config}
            classifier = classifier_cls(**model_args)

            # Get the parameters on the first and second fit with config params
            params_first = fitted_params(classifier)
            if hasattr(classifier.estimator, 'random_state'):
                rs_1 = classifier.random_state
                rs_estimator_1 = classifier.estimator.random_state

            params_second = fitted_params(classifier)
            if hasattr(classifier.estimator, 'random_state'):
                rs_2 = classifier.random_state
                rs_estimator_2 = classifier.estimator.random_state

            # An acceptable error occured, skip to next sample
            if params_first is None or params_second is None:
                continue

            # Remove keys we don't wish to include in the comparison
            for params in [params_first, params_second]:
                for key in param_keys_ignored:
                    if key in params:
                        del params[key]

            # They should have equal parameters
            self.assertEqual(params_first, params_second,
                             f"Failed with model args {model_args}")
            if hasattr(classifier.estimator, 'random_state'):
                assert all([
                    seed == random_state
                    for random_state in [rs_1, rs_estimator_1, rs_2, rs_estimator_2]
                ])
