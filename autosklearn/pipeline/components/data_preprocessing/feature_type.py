from typing import Any, List, Dict, Optional, Tuple, Union

import sklearn.compose
from scipy import sparse

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.base import BaseEstimator

from autosklearn.pipeline.base import (
     BasePipeline,
     DATASET_PROPERTIES_TYPE,
     PIPELINE_DATA_DTYPE,
 )
from autosklearn.pipeline.components.data_preprocessing.feature_type_categorical \
    import CategoricalPreprocessingPipeline
from autosklearn.pipeline.components.data_preprocessing.feature_type_numerical \
    import NumericalPreprocessingPipeline
from autosklearn.pipeline.components.base import AutoSklearnComponent, AutoSklearnChoice, \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT
from autosklearn.data.validation import (
    SUPPORTED_FEAT_TYPES,
    SUPPORTED_TARGET_TYPES,
)


class FeatTypeSplit(AutoSklearnPreprocessingAlgorithm):
    """ This component is used to apply distinct transformations to categorical and
    numerical features of a dataset. It is built on top of sklearn's ColumnTransformer.
    """

    def __init__(
        self,
        config: Optional[Configuration] = None,
        pipeline: Optional[BasePipeline] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict[str, Any]] = None,
        feat_type: Optional[Dict[Union[str, int], str]] = None,
        force_sparse_output: bool = False,
        column_transformer: Optional[sklearn.compose.ColumnTransformer] = None,
    ):

        if pipeline is not None:
            raise ValueError("DataPreprocessor's argument 'pipeline' should be None")

        self.config = config
        self.pipeline = pipeline
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude
        self.random_state = random_state
        self.init_params = init_params
        self.feat_type = feat_type
        self.force_sparse_output = force_sparse_output

        # The pipeline that will be applied to the categorical features (i.e. columns)
        # of the dataset
        # Configuration of the data-preprocessor is different from the configuration of
        # the categorical pipeline. Hence, force to None
        # It is actually the call to set_hyperparameter who properly sets this argument
        # TODO: Extract the child configuration space from the FeatTypeSplit to the
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
        # TODO: Extract the child configuration space from the FeatTypeSplit to the
        # pipeline if needed
        self.numer_ppl = NumericalPreprocessingPipeline(
            config=None, steps=pipeline, dataset_properties=dataset_properties,
            include=include, exclude=exclude, random_state=random_state,
            init_params=init_params)
        self._transformers: List[Tuple[str, AutoSklearnComponent]] = [
            ("categorical_transformer", self.categ_ppl),
            ("numerical_transformer", self.numer_ppl),
        ]
        if self.config:
            self.set_hyperparameters(self.config, init_params=init_params)
        self.column_transformer = column_transformer

    def fit(self, X: SUPPORTED_FEAT_TYPES, y: Optional[SUPPORTED_TARGET_TYPES] = None
            ) -> 'FeatTypeSplit':

        n_feats = X.shape[1]
        categorical_features = []
        numerical_features = []
        if self.feat_type is not None:
            # Make sure that we are not missing any column!
            expected = set(self.feat_type.keys())
            if hasattr(X, 'columns'):
                columns = set(X.columns)
            else:
                columns = set(range(n_feats))
            if expected != columns:
                raise ValueError("Train data has columns={} yet the feat_types are feat={}".format(
                    expected,
                    columns
                ))
            categorical_features = [key for key, value in self.feat_type.items()
                                    if value.lower() == 'categorical']
            numerical_features = [key for key, value in self.feat_type.items()
                                  if value.lower() == 'numerical']

        # If no categorical features, assume we have a numerical only pipeline
        if len(categorical_features) == 0:
            sklearn_transf_spec: List[Tuple[str, BaseEstimator, List[Union[str, bool, int]]]] = [
                ("numerical_transformer", self.numer_ppl, [True] * n_feats)
            ]
        # If all features are categorical, then just the categorical transformer is used
        elif len(numerical_features) == 0:
            sklearn_transf_spec = [
                ("categorical_transformer", self.categ_ppl, [True] * n_feats)
            ]
        # For the other cases, both transformers are used
        else:
            sklearn_transf_spec = [
                ("categorical_transformer", self.categ_ppl, categorical_features),
                ("numerical_transformer", self.numer_ppl, numerical_features)
            ]

        # And one last check in case feat type is None
        # And to make sure the final specification has all the columns
        # considered in the column transformer
        total_columns = sum([len(features) for name, ppl, features in sklearn_transf_spec])
        if total_columns != n_feats:
            raise ValueError("Missing columns in the specification of the data validator"
                             " for train data={} and spec={}".format(
                                 np.shape(X),
                                 sklearn_transf_spec,
                             ))

        self.sparse_ = sparse.issparse(X) or self.force_sparse_output
        self.column_transformer = sklearn.compose.ColumnTransformer(
            transformers=sklearn_transf_spec,
            sparse_threshold=float(self.sparse_),
            )
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X: SUPPORTED_FEAT_TYPES) -> PIPELINE_DATA_DTYPE:
        if self.column_transformer is None:
            raise ValueError("Cannot call transform on a Datapreprocessor that has not"
                             "yet been fit. Please check the log files for errors "
                             "while trying to fit the model."
                             )
        return self.column_transformer.transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'FeatTypeSplit',
                'name': 'Feature Type Splitter',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,), }

    def set_hyperparameters(self, configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None) -> 'FeatTypeSplit':
        if init_params is not None and 'feat_type' in init_params.keys():
            self.feat_type = init_params['feat_type']

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

            sub_init_params_dict: Optional[Dict[str, Any]] = None
            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % transf_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % transf_name, '', 1)
                        sub_init_params_dict[new_name] = value

            if isinstance(transf_op, (
                    AutoSklearnChoice, AutoSklearnComponent, BasePipeline)):
                transf_op.set_hyperparameters(
                    configuration=sub_configuration, init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(
        self,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        self.dataset_properties = dataset_properties
        cs = ConfigurationSpace()
        cs = FeatTypeSplit._get_hyperparameter_search_space_recursevely(
            dataset_properties, cs, self._transformers)
        return cs

    @staticmethod
    def _get_hyperparameter_search_space_recursevely(
        dataset_properties: DATASET_PROPERTIES_TYPE,
        cs: ConfigurationSpace,
        transformer: BaseEstimator,
    ) -> ConfigurationSpace:
        for st_name, st_operation in transformer:
            if hasattr(st_operation, "get_hyperparameter_search_space"):
                cs.add_configuration_space(
                    st_name,
                    st_operation.get_hyperparameter_search_space(dataset_properties))
            else:
                return FeatTypeSplit._get_hyperparameter_search_space_recursevely(
                    dataset_properties, cs, st_operation)
        return cs
