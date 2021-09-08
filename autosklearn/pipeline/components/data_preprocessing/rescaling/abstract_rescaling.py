from typing import Optional
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm

IGNORED_WARNINGS = [
    # The QuantileTransformerComponent is created before knowing the number of samples
    # and so the search space includes n_quantiles which can be too large
    (
        UserWarning,
        r'n_quantiles \(\d+\) is greater than the total number of samples \(\d+\)'
    )
]

class Rescaling(object):
    # Rescaling does not support fit_transform (as of 0.19.1)!
    def __init__(self, random_state: Optional[np.random.RandomState] = None):
        self.preprocessor: Optional[BaseEstimator] = None

    def fit(
        self,
        X: PIPELINE_DATA_DTYPE,
        y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> 'AutoSklearnPreprocessingAlgorithm':

        if self.preprocessor is None:
            raise NotFittedError()

        with warnings.catch_warnings():
            for category, message in IGNORED_WARNINGS:
                warnings.filterwarnings('ignore', category=category, message=message)

            self.preprocessor.fit(X)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:

        if self.preprocessor is None:
            raise NotFittedError()

        with warnings.catch_warnings():
            for category, message in IGNORED_WARNINGS:
                warnings.filterwarnings('ignore', category=category, message=message)

            transformed_X = self.preprocessor.transform(X)

        return transformed_X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
