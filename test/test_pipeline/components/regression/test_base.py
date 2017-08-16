import unittest

from autosklearn.pipeline.util import _test_regressor, \
    _test_regressor_iterative_fit

from autosklearn.pipeline.constants import *

import sklearn.metrics
import numpy as np


class BaseRegressionComponentTest(unittest.TestCase):

    res = None

    module = None
    sk_module = None

    # Magic command to not run tests on base class
    __test__ = False

    def test_default_boston(self):
        for i in range(2):
            predictions, targets = \
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
                self.assertAlmostEqual(self.res["default_boston"],
                                       sklearn.metrics.r2_score(targets,
                                                                predictions),
                                       places=self.res.get(
                                               "default_boston_places", 7))

    def test_default_boston_iterative_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets = \
                _test_regressor_iterative_fit(dataset="boston",
                                              Regressor=self.module)
            self.assertAlmostEqual(self.res["default_boston_iterative"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_boston_iterative_places", 7))

    def test_default_boston_iterative_sparse_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets = \
                _test_regressor_iterative_fit(dataset="boston",
                                              Regressor=self.module,
                                              sparse=True)
            self.assertAlmostEqual(self.res["default_boston_iterative_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_boston_iterative_sparse_places", 7))

    def test_default_boston_sparse(self):
        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets = \
                _test_regressor(dataset="boston",
                                Regressor=self.module,
                                sparse=True)
            self.assertAlmostEqual(self.res["default_boston_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_boston_sparse_places", 7))

    def test_default_diabetes(self):
        for i in range(2):
            predictions, targets = \
                _test_regressor(dataset="diabetes",
                                Regressor=self.module)

            self.assertAlmostEqual(self.res["default_diabetes"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_places", 7))

    def test_default_diabetes_iterative_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets = \
                _test_regressor_iterative_fit(dataset="diabetes",
                                              Regressor=self.module)
            self.assertAlmostEqual(self.res["default_diabetes_iterative"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_iterative_places", 7))

    def test_default_diabetes_iterative_sparse_fit(self):
        if not hasattr(self.module, 'iterative_fit'):
            return

        for i in range(2):
            predictions, targets = \
                _test_regressor_iterative_fit(dataset="diabetes",
                                              Regressor=self.module,
                                              sparse=True)
            self.assertAlmostEqual(self.res["default_diabetes_iterative_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_iterative_sparse_places", 7))

    def test_default_diabetes_sparse(self):
        if SPARSE not in self.module.get_properties()["input"]:
            return

        for i in range(2):
            predictions, targets = \
                _test_regressor(dataset="diabetes",
                                Regressor=self.module,
                                sparse=True)
            self.assertAlmostEqual(self.res["default_diabetes_sparse"],
                                   sklearn.metrics.r2_score(targets,
                                                            predictions),
                                   places=self.res.get(
                                           "default_diabetes_sparse_places", 7))