from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.exceptions import NotFittedError

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling import (  # noqa: E501
    Rescaling,
)
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class StandardScalerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        from sklearn.preprocessing import StandardScaler

        self.preprocessor = StandardScaler(copy=False)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "StandardScaler",
            "name": "StandardScaler",
            "handles_missing_values": False,
            "handles_nominal_values": False,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            # TODO find out if this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "AutoSklearnPreprocessingAlgorithm":
        if self.preprocessor is None:
            raise NotFittedError()
        if sparse.isspmatrix(X):
            self.preprocessor.set_params(with_mean=False)

        return super(StandardScalerComponent, self).fit(X, y)
