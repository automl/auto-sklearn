import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UnParametrizedHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool


class RidgeRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self, alpha, fit_intercept, tol, random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.linear_model

        self.alpha = float(self.alpha)
        self.fit_intercept = check_for_bool(self.fit_intercept)
        self.tol = float(self.tol)

        self.estimator = sklearn.linear_model.Ridge(alpha=self.alpha,
                                                    fit_intercept=self.fit_intercept,
                                                    tol=self.tol,
                                                    copy_X=True,
                                                    normalize=False,
                                                    random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Rigde',
                'name': 'Ridge Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'prefers_data_normalized': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            "alpha", 10 ** -5, 10., log=True, default_value=1.)
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1,
                                         default_value=1e-3, log=True)
        cs.add_hyperparameters([alpha, fit_intercept, tol])
        return cs
