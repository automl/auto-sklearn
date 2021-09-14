from typing import Dict, Optional, Tuple, Union

import numpy as np

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling \
    import Rescaling
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm


class PowerTransformerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        from sklearn.preprocessing import PowerTransformer
        self.preprocessor = PowerTransformer(copy=False)

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'PowerTransformer',
                'name': 'PowerTransformer',
                'handles_missing_values': False,
                'handles_nominal_values': False,
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
                'handles_sparse': False,
                'handles_dense': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}
