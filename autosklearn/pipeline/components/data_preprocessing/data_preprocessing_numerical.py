import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.data_preprocessing import rescaling as \
    rescaling_components
from autosklearn.pipeline.components.data_preprocessing.imputation.numerical_imputation \
    import NumericalImputation
from autosklearn.pipeline.components.data_preprocessing.variance_threshold\
    .variance_threshold import VarianceThreshold

from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


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

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`.

    """

    def __init__(self, config=None, steps=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None):
        self._output_dtype = np.int32
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'num_datapreproc',
                'name': 'numeric data preprocessing',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out if this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    def _get_hyperparameter_search_space(self, include=None, exclude=None,
                                         dataset_properties=None):
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
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        return cs

    def _get_pipeline_steps(self, dataset_properties=None):
        steps = []

        default_dataset_properties = {}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ["imputation", NumericalImputation()],
            ["variance_threshold", VarianceThreshold()],
            ["rescaling", rescaling_components.RescalingChoice(default_dataset_properties)],
            ])

        return steps

    def _get_estimator_hyperparameter_name(self):
        return "numerical data preprocessing"
