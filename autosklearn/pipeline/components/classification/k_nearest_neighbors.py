from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE


class KNearestNeighborsClassifier(AutoSklearnClassificationAlgorithm):

    def __init__(self, n_neighbors, weights, p, random_state=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.random_state = random_state

    def fit(self, X, Y):
        import sklearn.neighbors
        import sklearn.multiclass

        estimator = \
            sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                                   weights=self.weights,
                                                   p=self.p)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

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
    def get_properties(dataset_properties=None):
        return {'shortname': 'KNN',
                'name': 'K-Nearest Neighbor Classification',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add_hyperparameters([n_neighbors, weights, p])

        return cs
