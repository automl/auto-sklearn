import numpy as np

import sklearn.gaussian_process 
import sklearn.preprocessing

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter

from ParamSklearn.components.base import ParamSklearnRegressionAlgorithm
from ParamSklearn.constants import *


class GaussianProcess(ParamSklearnRegressionAlgorithm):
    def __init__(self, nugget, thetaL, thetaU, normalize=False, copy_X=False, 
                 random_state=None):
        self.nugget = float(nugget)
        self.thetaL = float(thetaL)
        self.thetaU = float(thetaU)
        self.normalize = normalize
        self.copy_X = copy_X
        # We ignore it
        self.random_state = random_state
        self.estimator = None
        self.scaler = None

    def fit(self, X, Y):
        # Instanciate a Gaussian Process model
        self.estimator = sklearn.gaussian_process.GaussianProcess(
            corr='squared_exponential', 
            theta0=np.ones(X.shape[1]) * 1e-1,
            thetaL=np.ones(X.shape[1]) * self.thetaL,
            thetaU=np.ones(X.shape[1]) * self.thetaU,
            nugget=self.nugget,
            optimizer='Welch',
            random_state=self.random_state)
        self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
        self.scaler.fit(Y)
        Y_scaled = self.scaler.transform(Y)
        self.estimator.fit(X, Y_scaled)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        if self.scaler is None:
            raise NotImplementedError
        Y_pred = self.estimator.predict(X, batch_size=512)
        return self.scaler.inverse_transform(Y_pred)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GP',
                'name': 'Gaussian Process',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        nugget = UniformFloatHyperparameter(
            name="nugget", lower=0.0001, upper=10, default=0.1, log=True)
        thetaL = UniformFloatHyperparameter(
            name="thetaL", lower=1e-6, upper=1e-3, default=1e-4, log=True)
        thetaU = UniformFloatHyperparameter(
            name="thetaU", lower=0.2, upper=10, default=1.0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(nugget)
        cs.add_hyperparameter(thetaL)
        cs.add_hyperparameter(thetaU)
        return cs
