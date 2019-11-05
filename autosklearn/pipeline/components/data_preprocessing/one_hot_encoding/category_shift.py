import numpy as np
from scipy import sparse
from sklearn.utils import check_array

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class CategoryShift(AutoSklearnPreprocessingAlgorithm):
    """ Add 3 to every category.
    Down in the pipeline, category 2 will be attribute to missing values,
    category 1 will be assigned to low occurence categories, and category 0
    is not used, so to provide compatibility with sparse matrices.
    """

    def __init__(self, random_state=None):
        self.random_stated = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Increment everything by three to account for the fact that
        # np.NaN will get an index of two, and coalesced values will get index of
        # one, index of zero is not assigned to also work with sparse data
        
        X_data = X.data if sparse.issparse(X) else X
        if np.nanmin(X_data) < 0:
            raise ValueError("X needs to contain only non-negative integers.")
        X_data += 3

        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CategShift',
                'name': 'Category Shift',
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
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        return ConfigurationSpace()
