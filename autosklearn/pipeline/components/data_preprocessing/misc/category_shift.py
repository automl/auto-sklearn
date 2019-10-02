import numpy as np
import scipy.sparse

import autosklearn.pipeline.implementations.CategoryShift

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class CategoryShift(AutoSklearnPreprocessingAlgorithm):
    def _fit(self, X, y=None):
        self.preprocessor = autosklearn.pipeline.implementations.CategoryShift\
            .CategoryShift()
        return self.preprocessor.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'cat_shift',
                'name': 'Category shift',
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
        cs = ConfigurationSpace()
        return cs
