from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.implementations.gem import GEM as GEMImpl
from autosklearn.pipeline.constants import *

class GEM(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, N, precond, random_state=None):
        self.N = N
        self.precond = precond

    def fit(self, X, Y):
        self.preprocessor = GEMImpl(self.N, self.precond)
        self.preprocessor.fit(X, Y)
        return self


    def transform(self, X):
        return self.preprocessor.transform(X)
    

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GEM',
                'name': 'Generalized Eigenvector extraction',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT, UNSIGNED_DATA)}


    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        N = UniformIntegerHyperparameter("N", 5, 20, default=10)
        precond = UniformFloatHyperparameter("precond", 0, 0.5, default=0.1)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(N)
        cs.add_hyperparameter(precond)
        return cs
