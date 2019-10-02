import numpy as np
import scipy.sparse


#from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class OneHotEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(self):
        pass

    def _fit(self, X, y=None):
        self.preprocessor = sklearn.preprocessing.OneHotEncoder(sparse=True)
        return self.preprocessor.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        is_sparse = scipy.sparse.issparse(X)
        X = self._fit(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

    def transform(self, X):
        is_sparse = scipy.sparse.issparse(X)
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

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
