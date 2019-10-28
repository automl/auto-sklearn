import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components.data_preprocessing.imputation.categorical_imputation \
    import CategoricalImputation
from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.minority_coalescer \
    import MinorityCoalescer
from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.one_hot_encoding \
    import OneHotEncoder


from autosklearn.pipeline.base import BasePipeline


class CategoricalPreprocessingPipeline(BasePipeline):
    """This class implements a pipeline for data preprocessing of categorical features.
    It assumes that the data to be transformed is made only of categorical features.
    The steps of this pipeline are:
        1 - Category Shift : Make sure that there are no categories with values 
            0, 1 and 2 in the dataset. These are special values, and they are used in 
            the next steps.
        2 - Imputation : Assign category 2 to missing values (NaN)
        3 - Minority coalescence: Assign category 1 to all categories whose occurence 
            don't sum-up to a certain minimum fraction
        4 - One hot encoding: traditional sklearn one hot encoding

    Parameters
    ----------

    Attributes
    ----------

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
        return {'shortname': 'cat_datapreproc',
                'name': 'categorical data preprocessing',
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
        
        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def _get_pipeline(self):
        steps = []

        steps.extend(
            [["imputation", CategoricalImputation()],
             ["category_coalescence", MinorityCoalescer()],
             ["one_hot_encoding", OneHotEncoder()]
             ])

        return steps

    def _get_estimator_hyperparameter_name(self):
        return "categorical data preprocessing"

