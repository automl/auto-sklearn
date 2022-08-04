from typing import Optional, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm


class Rescaling(object):
    # Rescaling does not support fit_transform (as of 0.19.1)!
    def __init__(
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.preprocessor: Optional[BaseEstimator] = None

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "AutoSklearnPreprocessingAlgorithm":

        if self.preprocessor is None:
            raise NotFittedError()

        self.preprocessor.fit(X)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:

        if self.preprocessor is None:
            raise NotFittedError()

        transformed_X = self.preprocessor.transform(X)

        return transformed_X

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
