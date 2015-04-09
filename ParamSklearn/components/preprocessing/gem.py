from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

from ParamSklearn.components.preprocessor_base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.implementations.gem import GEM as GEMImpl
from ParamSklearn.util import DENSE

class GEM(ParamSklearnPreprocessingAlgorithm):

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
    def get_properties():
        return {'shortname': 'GEM',
                'name': 'Generalized Eigenvector extraction',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': False,
                'handles_dense': True,
                'input': (DENSE, ),
                'output': DENSE,
                'preferred_dtype': None}


    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        N = UniformIntegerHyperparameter("N", 5, 20, default=10)
        precond = UniformFloatHyperparameter("precond", 0, 0.5, default=0.1)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(N)
        cs.add_hyperparameter(precond)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name

