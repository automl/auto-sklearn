from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import UniformIntegerHyperparameter

from ..preprocessor_base import AutoSklearnPreprocessingAlgorithm
from ...implementations.SparseFiltering import SparseFiltering as SparseFilteringImpl

class SparseFiltering(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, N, maxiter=100, random_state=None):
        self.N = N
        self.maxiter = maxiter
        self.random_state = random_state

    def fit(self, X, Y):
        self.preprocessor = SparseFilteringImpl(self.N, self.maxiter, random_state = self.random_state)
        self.preprocessor.fit(X, Y)
        return self
    
    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)
    
    @staticmethod
    def get_properties():
        return {'shortname': 'PCA',
                'name': 'Principle Component Analysis',
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
                'handles_sparse': False,
                'preferred_dtype': None}


    
    @staticmethod
    def get_hyperparameter_search_space():
        N = UniformIntegerHyperparameter(
            "N", 50, 2000, default=100)
        maxiter = UniformIntegerHyperparameter(
            "maxiter", 50, 500, default=100)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(N)
        cs.add_hyperparameter(maxiter)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "AutoSklearn %" % name
