from typing import Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np
from scipy.sparse import issparse

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class CategoricalImputation(AutoSklearnPreprocessingAlgorithm):
    """
    Substitute missing values by constant:
        When strategy == “constant”, fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing
        numerical data and “missing_value” for strings or object data types.
    """

    def __init__(
        self,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.random_state = random_state

    def fit(self, X: PIPELINE_DATA_DTYPE,
            y: Optional[PIPELINE_DATA_DTYPE] = None) -> 'CategoricalImputation':
        import sklearn.impute

        if hasattr(X, 'columns'):
            kind = X[X.columns[-1]].dtype.kind
        else:
            # Series, sparse and numpy have dtype
            # Only DataFrame does not
            kind = X.dtype.kind

        number_kinds = ("i", "u", "f")
        if kind in number_kinds:
            # We do not want to impute a category with the default
            # value (0 is the default).
            # Hence we take one greater than the max
            unique = np.unique([*X.data, 0]) if issparse(X) else np.unique(X)
            print(unique)
            fill_value = min(unique) - 1
        else:
            fill_value = None

        print(fill_value)

        self.preprocessor = sklearn.impute.SimpleImputer(
            strategy='constant', copy=False, fill_value=fill_value)
        self.preprocessor.fit(X)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'CategoricalImputation',
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
                'handles_multioutput': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        return ConfigurationSpace()
