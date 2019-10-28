import numpy as np
from scipy import sparse
from sklearn.utils import check_array

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class CategoricalImputation(AutoSklearnPreprocessingAlgorithm):
    """ Imputation of categorical features. It should be used as a first step on a data 
    preprocessing pipeline of categorical features. It makes sure categories are all
    integers greater or equal to three. Then missing values are substitued (imputed) by 
    the two. 
    
    """

    def __init__(self, random_state=None):
        self.random_stated = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Increment everything by three to account for the fact that
        # np.NaN will get an index of two, and coalesced values will get index of
        # one, index of zero is not assigned to also work with sparse data
        if sparse.issparse(X):
            X.data += 3
            X.data[~np.isfinite(X.data)] = 2
        else:
            X += 3
            X[~np.isfinite(X)] = 2

        X = check_array(X, accept_sparse='csc', force_all_finite=False,
                        dtype=np.int32)

        if X.min() < 0:
            raise ValueError("X needs to contain only non-negative integers.")

        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'CategImputation',
                'name': 'Categorical Imputation',
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
