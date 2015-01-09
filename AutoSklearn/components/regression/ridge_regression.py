import numpy as np
import sklearn.linear_model

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ..regression_base import AutoSklearnRegressionAlgorithm


class RidgeRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self, alpha, fit_intercept=False, normalize=False,
                 copy_X=False, max_iter=None, tol=0.001, solver='auto',
                 random_state=None):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        # We ignore it
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        self.estimator = sklearn.linear_model.Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver)

        return self.estimator.fit(X, Y)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'RR',
                'name': 'Ridge Regression',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': True,
                'is_deterministic': True,
                'handles_sparse': True,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space():
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0001, upper=10, default=1.0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        return cs
