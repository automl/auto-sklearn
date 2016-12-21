import numpy as np

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class GaussRandomProjection(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, eps, random_state=None):
        self.eps = eps
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.random_projection

        self.preprocessor = sklearn.random_projection.GaussianRandomProjection(
            eps=self.eps)
        self.preprocessor.fit(X, Y)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'random_projection',
                'name': 'Random Gaussian Projection',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (DENSE, INPUT)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        admissible_distortion = UniformFloatHyperparameter(
            "eps", 0.01, 1.0, default=0.5)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(admissible_distortion)
        return cs
