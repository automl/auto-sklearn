from HPOlibConfigSpace.configuration_space import ConfigurationSpace

from ParamSklearn.components.base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.constants import *


class NoPreprocessing(ParamSklearnPreprocessingAlgorithm):

    def __init__(self, random_state):
        """ This preprocessors does not change the data """
        self.preprocessor = None

    def fit(self, X, Y=None):
        self.preprocessor = 0
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'no',
                'name': 'NoPreprocessing',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name

