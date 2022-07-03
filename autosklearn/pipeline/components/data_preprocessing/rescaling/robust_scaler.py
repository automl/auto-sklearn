from typing import Dict, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from scipy import sparse
from sklearn.exceptions import NotFittedError

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling import (  # noqa: E501
    Rescaling,
)
from autosklearn.pipeline.constants import (
    DENSE,
    INPUT,
    SIGNED_DATA,
    SPARSE,
    UNSIGNED_DATA,
)


class RobustScalerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        q_min: float,
        q_max: float,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        from sklearn.preprocessing import RobustScaler

        self.q_min = q_min
        self.q_max = q_max
        self.preprocessor = RobustScaler(
            quantile_range=(self.q_min, self.q_max),
            copy=False,
        )

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "RobustScaler",
            "name": "RobustScaler",
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
            "output": (INPUT, SIGNED_DATA),
            "preferred_dtype": None,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        q_min = UniformFloatHyperparameter("q_min", 0.001, 0.3, default_value=0.25)
        q_max = UniformFloatHyperparameter("q_max", 0.7, 0.999, default_value=0.75)
        cs.add_hyperparameters((q_min, q_max))
        return cs

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "AutoSklearnPreprocessingAlgorithm":
        if self.preprocessor is None:
            raise NotFittedError()
        if sparse.isspmatrix(X):
            self.preprocessor.set_params(with_centering=False)

        return super(RobustScalerComponent, self).fit(X, y)
