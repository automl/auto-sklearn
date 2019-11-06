import numpy as np
import scipy.sparse

import sklearn.preprocessing

from ConfigSpace.configuration_space import ConfigurationSpace

import autosklearn.pipeline.implementations.SparseOneHotEncoder
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class OneHotEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        if scipy.sparse.issparse(X):
            self.preprocessor = autosklearn.pipeline.implementations.SparseOneHotEncoder\
                .SparseOneHotEncoder()
        else:
            self.preprocessor = sklearn.preprocessing\
                .OneHotEncoder(sparse=False, categories='auto')
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()
