import numpy as np

import sklearn.compose
from scipy import sparse

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.components.data_preprocessing.data_preprocessing_categorical \
    import CategoricalPreprocessingPipeline
from autosklearn.pipeline.components.data_preprocessing.data_preprocessing_numerical \
    import NumericalPreprocessingPipeline
from autosklearn.pipeline.components.base import AutoSklearnComponent, AutoSklearnChoice
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class DataPreprocessor(AutoSklearnComponent):
    """ This component is used to apply distinct transformations to categorical and
    numerical features of a dataset. It is built on top of sklearn's ColumnTransformer.
    """

    def __init__(self, config=None, pipeline=None, dataset_properties=None, include=None,
                 exclude=None, random_state=None, init_params=None,
                 categorical_features=None, force_sparse_output=False):

        if pipeline is not None:
            raise ValueError("DataPreprocessor's argument 'pipeline' should be None")

        if categorical_features is not None:
            categorical_features = np.array(categorical_features)
            if categorical_features.dtype != 'bool':
                raise ValueError('Parameter categorical_features must'
                                 ' only contain booleans.')
        self.categorical_features = categorical_features

        # The pipeline that will be applied to the categorical features (i.e. columns)
        # of the dataset
        self.categ_ppl = CategoricalPreprocessingPipeline(
            config, pipeline, dataset_properties, include, exclude,
            random_state, init_params)
        # The pipeline that will be applied to the numerical features (i.e. columns)
        # of the dataset
        self.numer_ppl = NumericalPreprocessingPipeline(
            config, pipeline, dataset_properties, include, exclude,
            random_state, init_params)
        self._transformers = [
            ["categorical_transformer", self.categ_ppl],
            ["numerical_transformer", self.numer_ppl],
        ]
        self.force_sparse = force_sparse_output

    def fit(self, X, y=None):
        # TODO: we are converting the categorical_features array from boolean flags
        # to integer indices to work around a sklearn bug. It should be fixed in sklearn
        # v0.22. Then we will be able to use the boolean array directly.

        n_feats = X.shape[1]
        # If categorical_features is none or an array made just of False booleans, then
        # only the numerical transformer is used
        numerical_features = np.logical_not(self.categorical_features)
        if self.categorical_features is None or np.all(numerical_features):
            sklearn_transf_spec = [
                ["numerical_transformer", self.numer_ppl, list(range(n_feats))]
            ]
        # If all features are categorical, then just the categorical transformer is used
        elif np.all(self.categorical_features):
            sklearn_transf_spec = [
                ["categorical_transformer", self.categ_ppl, list(range(n_feats))]
            ]
        # For the other cases, both transformers are used
        else:
            cat_feats = np.where(self.categorical_features)[0]
            num_feats = np.where(numerical_features)[0]
            sklearn_transf_spec = [
                ["categorical_transformer", self.categ_ppl, cat_feats],
                ["numerical_transformer", self.numer_ppl, num_feats]
            ]

        self.sparse_ = sparse.issparse(X) or self.force_sparse
        self.column_transformer = sklearn.compose.ColumnTransformer(
            transformers=sklearn_transf_spec,
            sparse_threshold=float(self.sparse_),
            )
        self.column_transformer.fit(X)
        return self

    def transform(self, X):
        if self.column_transformer is None:
            raise NotImplementedError()
        return self.column_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FeatTypeSplit',
                'name': 'Feature Type Splitter',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,), }

    def set_hyperparameters(self, configuration, init_params=None):
        if init_params is not None and 'categorical_features' in init_params.keys():
            self.categorical_features = init_params['categorical_features']

        self.configuration = configuration

        for transf_name, transf_op in self._transformers:
            sub_configuration_space = transf_op.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties_
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % transf_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % transf_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space,
                                              values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % transf_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % transf_name, '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(transf_op, (
                    AutoSklearnChoice, AutoSklearnComponent, BasePipeline)):
                transf_op.set_hyperparameters(
                    configuration=sub_configuration, init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self, dataset_properties=None):
        self.dataset_properties_ = dataset_properties
        cs = ConfigurationSpace()
        cs = DataPreprocessor._get_hyperparameter_search_space_recursevely(
            dataset_properties, cs, self._transformers)
        return cs

    @staticmethod
    def _get_hyperparameter_search_space_recursevely(dataset_properties, cs, transformer):
        for st_name, st_operation in transformer:
            if hasattr(st_operation, "get_hyperparameter_search_space"):
                cs.add_configuration_space(
                    st_name,
                    st_operation.get_hyperparameter_search_space(dataset_properties))
            else:
                return DataPreprocessor._get_hyperparameter_search_space_recursevely(
                    dataset_properties, cs, st_operation)
        return cs
