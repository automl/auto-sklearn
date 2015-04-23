import numpy as np
import sklearn.decomposition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant

from ParamSklearn.components.preprocessor_base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.util import SPARSE, DENSE, INPUT


class DictionaryLearning(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, n_components, alpha, max_iter, tol, fit_algorithm,
                 transform_algorithm, transform_alpha, split_sign,
                 random_state=None):
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_alpha = bool(transform_alpha)
        self.split_sign = bool(split_sign)
        self.random_state = random_state

    def fit(self, X, Y=None):
        self.preprocessor = sklearn.decomposition.DictionaryLearning(
            n_components=self.n_components, alpha=self.alpha,
            max_iter=self.max_iter, tol=self.tol,
            fit_algorithm=self.fit_algorithm,
            transform_algorithm=self.transform_algorithm,
            transform_alpha=self.transform_alpha,
            split_sign=self.split_sign, random_state=self.random_state
        )
        X = X.astype(np.float64)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'Dictionary Learning',
                'name': 'Dictionary Learning',
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
            "n_components", 50, 2000, default=100)
        alpha = UniformFloatHyperparameter(
            "alpha", 1e-5, 10, 1, log=True)
        max_iter = UniformIntegerHyperparameter(
            "max_iter", 50, 500, default=100)
        tol = UniformFloatHyperparameter('tol', 1e-9, 1e-3, 1e-8, log=True)
        fit_algorithm = CategoricalHyperparameter('fit_algorithm',
                                                  ['lars', 'cd'], 'lars')
        transform_algorithm = CategoricalHyperparameter('transform_algorithm',
            ['lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'], 'omp')
        transform_alpha = UniformFloatHyperparameter('transform_alpha',
            0.1, 10., 1., log=True)
        split_sign = CategoricalHyperparameter('split_sign', ['False',
                                                              'True'], 'False')
        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_components)
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(max_iter)
        cs.add_hyperparameter(tol)
        cs.add_hyperparameter(fit_algorithm)
        cs.add_hyperparameter(transform_algorithm)
        cs.add_hyperparameter(transform_alpha)
        cs.add_hyperparameter(split_sign)
        return cs


