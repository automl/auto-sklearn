from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, UNSIGNED_DATA, DENSE, INPUT


class Densifier(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        pass

    def fit(self, X, y=None):
        self.fitted_ = True
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
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (SPARSE, UNSIGNED_DATA),
                'output': (DENSE, INPUT)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
