from typing import Dict, Optional, Tuple, Union


from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class NoCoalescence(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        pass

    def fit(self, X: np.array, y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> PIPELINE_DATA_DTYPE:
        self.preprocessor = 'passthrough'
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return X

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
