from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class KNearestNeighborsRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_neighbors, weights, p, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state

    def fit(self, X, Y):
        import sklearn.neighbors

        self.estimator = \
            sklearn.neighbors.KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                p=self.p)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KNN',
                'name': 'K-Nearest Neighbor Classification',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_neighbors = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default=1))
        weights = cs.add_hyperparameter(CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default="uniform"))
        p = cs.add_hyperparameter(CategoricalHyperparameter(
            name="p", choices=[1, 2], default=2))

        return cs
