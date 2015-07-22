import warnings

import sklearn.decomposition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from HPOlibConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause

from ParamSklearn.components.base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.util import SPARSE, DENSE, INPUT

import numpy as np


class FastICA(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, n_components, algorithm, whiten, fun,
                 random_state=None):
        self.n_components = int(n_components)
        self.algorithm = algorithm
        self.whiten = bool(whiten)
        self.fun = fun
        self.random_state = random_state

    def fit(self, X, Y=None):
        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components, algorithm=self.algorithm,
            fun=self.fun, whiten=self.whiten, random_state=self.random_state
        )
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, ),
                'output': INPUT,
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default=100)
        algorithm = CategoricalHyperparameter('algorithm',
            ['parallel', 'deflation'], 'parallel')
        whiten = CategoricalHyperparameter('whiten',
            ['False', 'True'], 'False')
        fun = CategoricalHyperparameter('fun', ['logcosh', 'exp', 'cube'],
                                        'logcosh')
        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_components)
        cs.add_hyperparameter(algorithm)
        cs.add_hyperparameter(whiten)
        cs.add_hyperparameter(fun)
        return cs


