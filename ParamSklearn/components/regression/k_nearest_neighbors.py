import sklearn.neighbors

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    Constant, UniformIntegerHyperparameter

from ParamSklearn.components.base import ParamSklearnRegressionAlgorithm
from ParamSklearn.constants import *


class KNearestNeighborsRegressor(ParamSklearnRegressionAlgorithm):
    def __init__(self, n_neighbors, weights, p, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state

    def fit(self, X, Y):
        self.estimator = \
            sklearn.neighbors.KNeighborsClassifier(
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
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_neighbors = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, default=1))
        weights = cs.add_hyperparameter(CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default="uniform"))
        p = cs.add_hyperparameter(CategoricalHyperparameter(
            name="p", choices=[1, 2], default=2))

        return cs
