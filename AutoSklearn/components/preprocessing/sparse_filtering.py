from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import UniformIntegerHyperparameter

from ...implementations.SparseFiltering import SparseFiltering

class SparseFiltering(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, N, maxiter=200):
        self.N = N
        self.maxiter = maxiter

    def fit(self, X, Y):
        self.preprocessor = SparseFiltering(self.N, self.maxiter)
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
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False,
                'handles_sparse': False,
                'preferred_dtype': None}


    
    @staticmethod
    def get_hyperparameter_search_space():
        N = UniformIntegerHyperparameter(
            "N", 100, 1000, default=200)
        maxiter = UniformIntegerHyperparameter(
            "maxiter", 50, 500, default=200)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(N)
        cs.add_hyperparameter(maxiter)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "AutoSklearn %" % name
