import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class TruncatedSVD(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, target_dim, random_state=None):
        self.target_dim = target_dim
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.decomposition

        self.target_dim = int(self.target_dim)
        target_dim = min(self.target_dim, X.shape[1] - 1)
        self.preprocessor = sklearn.decomposition.TruncatedSVD(
            target_dim, algorithm='randomized')
        # TODO: remove when migrating to sklearn 0.16
        # Circumvents a bug in sklearn
        # https://github.com/scikit-learn/scikit-learn/commit/f08b8c8e52663167819f242f605db39f3b5a6d0c
        # X = X.astype(np.float64)
        self.preprocessor.fit(X, Y)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'TSVD',
                'name': 'Truncated Singular Value Decomposition',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (SPARSE, UNSIGNED_DATA),
                'output': (DENSE, INPUT)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        target_dim = UniformIntegerHyperparameter(
            "target_dim", 10, 256, default_value=128)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(target_dim)
        return cs
