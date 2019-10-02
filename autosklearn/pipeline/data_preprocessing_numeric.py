import copy
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from autosklearn.pipeline.components.data_preprocessing import rescaling as \
    rescaling_components
from autosklearn.pipeline.components.data_preprocessing.imputation.imputation \
    import Imputation
from autosklearn.pipeline.components.data_preprocessing.variance_threshold.variance_threshold \
    import VarianceThreshold

from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.constants import *


class NumericPreprocessingPipeline(BasePipeline):
    """This class implements the classification task.

    It implements a pipeline, which includes one preprocessing step and one
    classification algorithm. It can render a search space including all known
    classification and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    ConfigSpace.configuration_space.Configuration.

    Parameters
    ----------
    configuration : ConfigSpace.configuration_space.Configuration
        The configuration to evaluate.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    _estimator : The underlying scikit-learn classification model. This
        variable is assigned after a call to the
        :meth:`autosklearn.pipeline.classification.SimpleClassificationPipeline
        .fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`autosklearn.pipeline.classification.SimpleClassificationPipeline
        .fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def __init__(self, config=None, pipeline=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None):
        self._output_dtype = np.int32
        super().__init__(
            config, pipeline, dataset_properties, include, exclude,
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



    def fit_transformer(self, X, y, fit_params=None):

        if fit_params is None:
            fit_params = {}

        X, fit_params = super().fit_transformer(
            X, y, fit_params=fit_params)

        return X, fit_params


    def _get_hyperparameter_search_space(self, include=None, exclude=None,
                                         dataset_properties=None):
        """Create the hyperparameter configuration space.

        Parameters
        ----------
        include : dict (optional, default=None)

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if not 'target_type' in dataset_properties:
            dataset_properties['target_type'] = 'classification'
        if dataset_properties['target_type'] != 'classification':
            dataset_properties['target_type'] = 'classification'

        if 'sparse' not in dataset_properties:
            # This dataset is probably dense
            dataset_properties['sparse'] = False

        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def _get_pipeline(self):
        steps = []

        default_dataset_properties = {'target_type': 'classification'}

        # Add the always active preprocessing components

        steps.extend(
            [["imputation", Imputation()],
             ["variance_threshold", VarianceThreshold()],
             ["rescaling",
              rescaling_components.RescalingChoice(default_dataset_properties)]
            ])

        return steps

    def _get_estimator_hyperparameter_name(self):
        return "data preprocessing"

