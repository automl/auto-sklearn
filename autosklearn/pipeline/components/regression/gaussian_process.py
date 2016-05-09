import numpy as np
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class GaussianProcess(AutoSklearnRegressionAlgorithm):
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
        import sklearn.gaussian_process
        import sklearn.preprocessing

        # Instanciate a Gaussian Process model
        self.estimator = sklearn.gaussian_process.GaussianProcess(
            corr='squared_exponential', 
            theta0=np.ones(X.shape[1]) * 1e-1,
            thetaL=np.ones(X.shape[1]) * self.thetaL,
            thetaU=np.ones(X.shape[1]) * self.thetaU,
            nugget=self.nugget,
            optimizer='Welch',
            random_state=self.random_state)

        # Remove this in sklearn==0.18 as the GP class will be refactored and
        #  hopefully not be affected by this problem any more.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
            self.scaler.fit(Y.reshape((-1, 1)))
            Y_scaled = self.scaler.transform(Y.reshape((-1, 1))).ravel()
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
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

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
