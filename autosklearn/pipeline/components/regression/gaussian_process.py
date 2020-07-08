from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS


class GaussianProcess(AutoSklearnRegressionAlgorithm):
    def __init__(self, alpha, thetaL, thetaU, random_state=None):
        self.alpha = alpha
        self.thetaL = thetaL
        self.thetaU = thetaU
        # We ignore it
        self.random_state = random_state
        self.estimator = None
        self.scaler = None

    def fit(self, X, y):
        import sklearn.gaussian_process

        self.alpha = float(self.alpha)
        self.thetaL = float(self.thetaL)
        self.thetaU = float(self.thetaU)

        n_features = X.shape[1]
        kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=[1.0]*n_features,
            length_scale_bounds=[(self.thetaL, self.thetaU)]*n_features)

        # Instanciate a Gaussian Process model
        self.estimator = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer='fmin_l_bfgs_b',
            alpha=self.alpha,
            copy_X_train=True,
            random_state=self.random_state,
            normalize_y=True)

        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GP',
                'name': 'Gaussian Process',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True)
        thetaL = UniformFloatHyperparameter(
            name="thetaL", lower=1e-10, upper=1e-3, default_value=1e-6, log=True)
        thetaU = UniformFloatHyperparameter(
            name="thetaU", lower=1.0, upper=100000, default_value=100000.0, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([alpha, thetaL, thetaU])
        return cs
