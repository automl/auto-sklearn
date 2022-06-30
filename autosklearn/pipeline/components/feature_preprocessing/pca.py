from typing import Optional

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from autosklearn.askl_typing import (
    DATASET_PROPERTIES_TYPE,
    FEAT_TYPE_TYPE,
    RANDOM_STATE_TYPE,
)
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA
from autosklearn.util.common import check_for_bool


class PCA(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self, keep_variance, whiten, random_state: Optional[RANDOM_STATE_TYPE] = None
    ):
        self.keep_variance = keep_variance
        self.whiten = whiten

        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.decomposition

        n_components = float(self.keep_variance)
        self.whiten = check_for_bool(self.whiten)

        self.preprocessor = sklearn.decomposition.PCA(
            n_components=n_components, whiten=self.whiten, copy=True
        )
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None):
        return {
            "shortname": "PCA",
            "name": "Principle Component Analysis",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            # TODO document that we have to be very careful
            "is_deterministic": False,
            "input": (DENSE, UNSIGNED_DATA),
            "output": (DENSE, UNSIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ):
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 0.9999, default_value=0.9999
        )
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default_value="False"
        )
        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten])
        return cs
