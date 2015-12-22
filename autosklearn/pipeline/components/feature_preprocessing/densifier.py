from HPOlibConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class Densifier(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy import sparse
        if sparse.issparse(X):
            return X.todense().getA()
        else:
            return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': False,
                'input': (SPARSE, UNSIGNED_DATA),
                'output': (DENSE, INPUT),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

