from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, BasePipeline
from autosklearn.pipeline.components.data_preprocessing import (
    rescaling as rescaling_components,
)
from autosklearn.pipeline.components.data_preprocessing.imputation.numerical_imputation import (  # noqa: E501
    NumericalImputation,
)
from autosklearn.pipeline.components.data_preprocessing.variance_threshold.variance_threshold import (  # noqa: E501
    VarianceThreshold,
)
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class NumericalPreprocessingPipeline(BasePipeline):
    """This class implements a pipeline for data preprocessing of numerical features.
    It assumes that the data to be transformed is made only of numerical features.
    The steps of this pipeline are:
        1 - Imputation: Substitution of missing values (NaN)
        2 - VarianceThreshold: Removes low-variance features
        3 - Rescaling: rescale features according to a certain rule (e.g. normalization,
            standartization or min-max)
    Parameters
    ----------
    config : ConfigSpace.configuration_space.Configuration
        The configuration to evaluate.
    random_state : Optional[int | RandomState]
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`."""

    def __init__(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, BaseEstimator]]] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._output_dtype = np.int32
        super().__init__(
            config=config,
            steps=steps,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
            random_state=random_state,
            init_params=init_params,
            feat_type=feat_type,
        )

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "num_datapreproc",
            "name": "numeric data preprocessing",
            "handles_missing_values": True,
            "handles_nominal_values": True,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "is_deterministic": True,
            # TODO find out if this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }

    def _get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        """Create the hyperparameter configuration space.
        Parameters
        ----------
        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()

        cs = self._get_base_search_space(
            cs=cs,
            dataset_properties=dataset_properties,
            exclude=exclude,
            include=include,
            pipeline=self.steps,
            feat_type=feat_type,
        )

        return cs

    def _get_pipeline_steps(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, BaseEstimator]]:
        steps = []

        default_dataset_properties = {}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend(
            [
                ("imputation", NumericalImputation(random_state=self.random_state)),
                (
                    "variance_threshold",
                    VarianceThreshold(random_state=self.random_state),
                ),
                (
                    "rescaling",
                    rescaling_components.RescalingChoice(
                        feat_type=feat_type,
                        dataset_properties=default_dataset_properties,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        return steps

    def _get_estimator_hyperparameter_name(self) -> str:
        return "numerical data preprocessing"
