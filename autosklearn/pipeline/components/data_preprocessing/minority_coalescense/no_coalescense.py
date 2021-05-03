from typing import Dict, Optional, Tuple, Union


from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class NoCoalescence(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        pass

    def fit(self, X: np.array, y: Optional[np.ndarray] = None
            ) -> np.ndarray:
        self.preprocessor = 'passthrough'
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'no coalescence',
                'name': 'No categorical variable coalescence',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
