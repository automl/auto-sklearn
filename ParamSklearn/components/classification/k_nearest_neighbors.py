import sklearn.neighbors

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, PREDICTIONS


class KNearestNeighborsClassifier(ParamSklearnClassificationAlgorithm):

    def __init__(self, n_neighbors, weights, metric, algorithm='auto', p=2,
                 leaf_size=30, random_state=None):

        self.n_neighbors = int(n_neighbors)
        if weights not in ("uniform", "distance"):
            raise ValueError("'weights' should be in ('uniform', 'distance'): "
                             "%s" % weights)
        self.weights = weights
        if metric not in ("euclidean", "manhattan", "chebyshev", "minkowski"):
            raise ValueError("'metric' should be in ('euclidean', 'chebyshev', "
                             "'manhattan', 'minkowski'): %s" % metric)
        self.metric = metric
        self.algorithm = algorithm
        self.p = int(p)
        self.leaf_size = int(leaf_size)
        self.random_state = random_state

    def fit(self, X, Y):
        self.estimator = \
            sklearn.neighbors.KNeighborsClassifier()
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'KNN',
                'name': 'K-Nearest Neighbor Classification',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (DENSE, ),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                'preferred_dtype' : None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, default=1)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default="uniform")
        metric = UnParametrizedHyperparameter(name="metric", value="minkowski")
        algorithm = Constant(name='algorithm', value="auto")
        p = CategoricalHyperparameter(
            name="p", choices=[1, 2, 5], default=2)
        leaf_size = Constant(name="leaf_size", value=30)

        # Unparametrized
        # TODO: If we further parametrize 'metric' we need more metric params
        metric = UnParametrizedHyperparameter(name="metric", value="minkowski")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_neighbors)
        cs.add_hyperparameter(weights)
        cs.add_hyperparameter(metric)
        cs.add_hyperparameter(algorithm)
        cs.add_hyperparameter(p)
        cs.add_hyperparameter(leaf_size)

        # Conditions
        metric_p = EqualsCondition(parent=metric, child=p, value="minkowski")
        cs.add_condition(metric_p)

        return cs
