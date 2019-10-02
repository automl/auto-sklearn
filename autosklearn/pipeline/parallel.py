import copy
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.constants import SPARSE


class ParallelPipeline(ClassifierMixin, BasePipeline):
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

    def __init__(self, pipeline_A, pipeline_B, config=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None):
        
        #self._output_dtype = np.int32

        self.pipeline_A = pipeline_A
        self.pipeline_B = pipeline_B

        super().__init__(
            config, None, dataset_properties, include, exclude,
            random_state, init_params)

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

        """
        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if not 'target_type' in dataset_properties:
            dataset_properties['target_type'] = 'classification'
        if dataset_properties['target_type'] != 'classification':
            dataset_properties['target_type'] = 'classification'

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        """

        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def _get_pipeline(self):
        #steps = [
        #    ['pipe_A', self.pipeline_A], 
        #    ['pipe_B', self.pipeline_B]]

        steps = [
            ['parallel_pipe', ColumnTransformer([
                ('pipe_A', self.pipeline_A, cf),
                ('pipe_B', self.pipeline_B, nf)])
            ]]
        
        return steps

    def _get_estimator_hyperparameter_name(self):
        return "classifier"

