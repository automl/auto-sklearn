from typing import Dict, Optional, Tuple, Union
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class NoEncoding(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None
            ) -> 'NoEncoding':
        self.preprocessor = 'passthrough'
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'no encoding',
                'name': 'No categorical variable encoding',
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
