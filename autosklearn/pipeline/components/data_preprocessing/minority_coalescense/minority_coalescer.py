from typing import Dict, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import autosklearn.pipeline.implementations.MinorityCoalescer
from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class MinorityCoalescer(AutoSklearnPreprocessingAlgorithm):
    """Group categories whose occurence is less than a specified minimum fraction."""

    def __init__(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        minimum_fraction: float = 0.01,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.minimum_fraction = minimum_fraction

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "MinorityCoalescer":
        self.minimum_fraction = float(self.minimum_fraction)

        self.preprocessor = (
            autosklearn.pipeline.implementations.MinorityCoalescer.MinorityCoalescer(
                minimum_fraction=self.minimum_fraction
            )
        )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "coalescer",
            "name": "Categorical minority coalescer",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            # TODO find out of this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        minimum_fraction = UniformFloatHyperparameter(
            "minimum_fraction", lower=0.0001, upper=0.5, default_value=0.01, log=True
        )
        cs.add_hyperparameter(minimum_fraction)
        return cs
