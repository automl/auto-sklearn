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
                 categorical_features=None, force_sparse_output=False,
                 column_transformer=None):

        if pipeline is not None:
            raise ValueError("DataPreprocessor's argument 'pipeline' should be None")

        if categorical_features is not None:
            categorical_features = np.array(categorical_features)
            if categorical_features.dtype != 'bool':
                raise ValueError('Parameter categorical_features must'
                                 ' only contain booleans.')
        self.config = config
        self.pipeline = pipeline
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude
        self.random_state = random_state
        self.init_params = init_params
        self.categorical_features = categorical_features
        self.force_sparse_output = force_sparse_output

        # The pipeline that will be applied to the categorical features (i.e. columns)
        # of the dataset
        # Configuration of the data-preprocessor is different from the configuration of
        # the categorical pipeline. Hence, force to None
        # It is actually the call to set_hyperparameter who properly sets this argument
        # TODO: Extract the child configuration space from the datapreprocessor to the
        # pipeline if needed
        self.categ_ppl = CategoricalPreprocessingPipeline(
            config=None, steps=pipeline, dataset_properties=dataset_properties,
            include=include, exclude=exclude, random_state=random_state,
            init_params=init_params)
        # The pipeline that will be applied to the numerical features (i.e. columns)
        # of the dataset
        # Configuration of the data-preprocessor is different from the configuration of
        # the numerical pipeline. Hence, force to None
        # It is actually the call to set_hyperparameter who properly sets this argument
        # TODO: Extract the child configuration space from the datapreprocessor to the
        # pipeline if needed
        self.numer_ppl = NumericalPreprocessingPipeline(
            config=None, steps=pipeline, dataset_properties=dataset_properties,
            include=include, exclude=exclude, random_state=random_state,
            init_params=init_params)
        self._transformers = [
            ["categorical_transformer", self.categ_ppl],
            ["numerical_transformer", self.numer_ppl],
        ]
        if self.config:
            self.set_hyperparameters(self.config, init_params=init_params)
        self.column_transformer = column_transformer

    def fit(self, X, y=None):

        n_feats = X.shape[1]
        # If categorical_features is none or an array made just of False booleans, then
        # only the numerical transformer is used
        numerical_features = np.logical_not(self.categorical_features)
        if self.categorical_features is None or np.all(numerical_features):
            sklearn_transf_spec = [
                ["numerical_transformer", self.numer_ppl, [True] * n_feats]
            ]
        # If all features are categorical, then just the categorical transformer is used
        elif np.all(self.categorical_features):
            sklearn_transf_spec = [
                ["categorical_transformer", self.categ_ppl, [True] * n_feats]
            ]
        # For the other cases, both transformers are used
        else:
            cat_feats = self.categorical_features
            num_feats = np.logical_not(self.categorical_features)
            sklearn_transf_spec = [
                ["categorical_transformer", self.categ_ppl, cat_feats],
                ["numerical_transformer", self.numer_ppl, num_feats]
            ]

        self.sparse_ = sparse.issparse(X) or self.force_sparse_output
        self.column_transformer = sklearn.compose.ColumnTransformer(
            transformers=sklearn_transf_spec,
            sparse_threshold=float(self.sparse_),
            )
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.column_transformer is None:
            raise ValueError("Cannot call transform on a Datapreprocessor that has not"
                             "yet been fit. Please check the log files for errors "
                             "while trying to fit the model."
                             )
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

        self.config = configuration

        for transf_name, transf_op in self._transformers:
            sub_configuration_space = transf_op.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties
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
        self.dataset_properties = dataset_properties
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
