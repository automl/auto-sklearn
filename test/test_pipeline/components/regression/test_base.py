from typing import Dict, Optional, Tuple

import unittest

import numpy as np
import sklearn.metrics

from autosklearn.pipeline.util import _test_regressor, \
    _test_regressor_iterative_fit
from autosklearn.pipeline.constants import SPARSE


class BaseRegressionComponentTest(unittest.TestCase):

    res = None

    module = None
    sk_module = None
    # Hyperparameter which is increased by iterative_fit
    step_hyperparameter = None

    # Magic command to not run tests on base class
    __test__ = False

    def test_default_boston(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        for i in range(2):
            predictions, targets, n_calls = \
                _test_regressor(dataset="boston",
                                Regressor=self.module)

            if "default_boston_le_ge" in self.res:
                # Special treatment for Gaussian Process Regression
                self.assertLessEqual(
                    sklearn.metrics.r2_score(y_true=targets,
                                             y_pred=predictions),
                    self.res["default_boston_le_ge"][0])
                self.assertGreaterEqual(
                    sklearn.metrics.r2_score(y_true=targets,
                                             y_pred=predictions),
                    self.res["default_boston_le_ge"][1])
            else:
                score = sklearn.metrics.r2_score(targets, predictions)
                fixture = self.res["default_boston"]
                if score < -1e10:
                    score = np.log(-score)
                    fixture = np.log(-fixture)
                self.assertAlmostEqual(
                    fixture,
                    score,
                    places=self.res.get("default_boston_places", 7),
                )

            if self.res.get("boston_n_calls"):
                self.assertEqual(self.res["boston_n_calls"], n_calls)

    def test_default_boston_iterative_fit(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets, regressor = \
                _test_regressor_iterative_fit(dataset="boston",
                                              Regressor=self.module)
            score = sklearn.metrics.r2_score(targets, predictions)
            fixture = self.res["default_boston_iterative"]

            if score < -1e10:
                score = np.log(-score)
                fixture = np.log(-fixture)

            self.assertAlmostEqual(
                fixture,
                score,
                places=self.res.get("default_boston_iterative_places", 7),
            )

            if self.step_hyperparameter is not None:
                self.assertEqual(
                    getattr(regressor.estimator, self.step_hyperparameter['name']),
                    self.res.get("boston_iterative_n_iter", self.step_hyperparameter['value'])
                )

    def test_default_boston_iterative_sparse_fit(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        if not hasattr(self.module, 'iterative_fit'):
            return

        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets, _ = \
                _test_regressor_iterative_fit(dataset="boston",
                                              Regressor=self.module,
                                              sparse=True)
            self.assertAlmostEqual(self.res["default_boston_iterative_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_boston_iterative_sparse_places", 7))

    def test_default_boston_sparse(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets, _ = \
                _test_regressor(dataset="boston",
                                Regressor=self.module,
                                sparse=True)
            self.assertAlmostEqual(self.res["default_boston_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_boston_sparse_places", 7))

    def test_default_diabetes(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        for i in range(2):
            predictions, targets, n_calls = \
                _test_regressor(dataset="diabetes",
                                Regressor=self.module)

            self.assertAlmostEqual(self.res["default_diabetes"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_places", 7))

            if self.res.get("diabetes_n_calls"):
                self.assertEqual(self.res["diabetes_n_calls"], n_calls)

    def test_default_diabetes_iterative_fit(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets, _ = \
                _test_regressor_iterative_fit(dataset="diabetes",
                                              Regressor=self.module)
            self.assertAlmostEqual(self.res["default_diabetes_iterative"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_iterative_places", 7))

    def test_default_diabetes_iterative_sparse_fit(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        if not hasattr(self.module, 'iterative_fit'):
            return

        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets, regressor = \
                _test_regressor_iterative_fit(dataset="diabetes",
                                              Regressor=self.module,
                                              sparse=True)
            self.assertAlmostEqual(self.res["default_diabetes_iterative_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_iterative_sparse_places", 7))

            if self.step_hyperparameter is not None:
                self.assertEqual(
                    getattr(regressor.estimator, self.step_hyperparameter['name']),
                    self.res.get("diabetes_iterative_n_iter", self.step_hyperparameter['value'])
                )

    def test_default_diabetes_sparse(self):

        if self.__class__ == BaseRegressionComponentTest:
            return

        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets, _ = \
                _test_regressor(dataset="diabetes",
                                Regressor=self.module,
                                sparse=True)
            self.assertAlmostEqual(self.res["default_diabetes_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_sparse_places", 7))

    def test_module_idempotent(self):
        """ Fitting twice with the same config gives the same model params. """
        if self.__class__ == BaseRegressionComponentTest:
            return

        regressor_cls = self.module

        X = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.5], [1.0, 1.0]])
        y = np.array([1, 1, 1, 1])

        # There are certain errors we ignore so we wrap this in a function
        def fitted_params(model) -> Dict:
            """
            Returns the params if fitted successfully, else None if an
            acceptable error occurs

            Currently we have no acceptable errors tracked for this regressors
            """
            model.fit(X.copy(), y.copy())
            return model.estimator.get_params()

        def random_state_keys(model) -> Optional[Tuple]:
            """ Gets the keys used for random number generation """
            if model.estimator is None:
                return None  # An acceptable error occured in fitting
            else:
                return model.estimator.random_state.get_state()[1]

        # We ignore certain keys when comparing
        # Random state objects don't compare so we manually compare it's keys
        param_keys_ignored = [
            'random_state', 'base_estimator', *self.res.get('ignore_hps', [])
        ]

        # We use the default config + sampled ones
        configuration_space = regressor_cls.get_hyperparameter_search_space()

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
            regressor = regressor_cls(**model_args)

            # Get the parameters on the first and second fit with config params
            # We also ensure their random state has been correctly reset such
            # that the state of their RandomState is the same after each fit call
            params_first = fitted_params(regressor)
            rand_state_keys_first = random_state_keys(regressor)

            params_second = fitted_params(regressor)
            rand_state_keys_second = random_state_keys(regressor)

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

            # Their random state keys (np.ndarray) should be the same
            assert all(rand_state_keys_first == rand_state_keys_second), \
                             f"The random states for {self.module} changed")

