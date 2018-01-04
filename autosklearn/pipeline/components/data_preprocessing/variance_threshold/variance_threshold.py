from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *

import sklearn.feature_selection


class VarianceThreshold(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        # VarianceThreshold does not support fit_transform (as of 0.19.1)!
        pass

    def fit(self, X, y=None):
        self.preprocessor = sklearn.feature_selection.VarianceThreshold(
            threshold=0.0
        )
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Variance Threshold',
            'name': 'Variance Threshold (constant feature removal)',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'is_deterministic': True,
            'handles_sparse': True,
            'handles_dense': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
