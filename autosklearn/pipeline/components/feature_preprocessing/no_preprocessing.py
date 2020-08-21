from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, random_state):
        """ This preprocessors does not change the data """
        self.preprocessor = None

    def fit(self, X, Y=None):
        self.preprocessor = 0
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'no',
                'name': 'NoPreprocessing',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
