from typing import Dict, Union, Optional
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA

DATASET_PROPERTIES_TYPE = Dict[str, Union[str, int, bool]]


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, random_state):
        """This preprocessors does not change the data"""

    def fit(self, X, Y=None):
        self.preprocessor = "passthrough"
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "no",
            "name": "NoPreprocessing",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type: Optional[Dict[Union[str, int], str]] = None,
            dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
    ):
        cs = ConfigurationSpace()
        return cs
