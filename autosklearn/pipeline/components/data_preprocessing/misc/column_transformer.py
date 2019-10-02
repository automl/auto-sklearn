import numpy as np
import scipy.sparse

import sklearn.compose


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none


class ColumnTransformer(AutoSklearnComponent):
    def __init__(self, transformers):
        self.transformers = transformers
        

    def _fit(self, X, y=None):
        self.column_transformer = sklearn.compose.ColumnTransformer(self.transformers)
        return self.column_transformer.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.column_transformer is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'columntransformer',
                'name': 'Column Transformer',
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
        for transf in self.transformers:
            transf.get_hyperparameter_search_space()
        return cs
