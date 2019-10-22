import numpy as np
import scipy.sparse

import sklearn.compose
from sklearn.pipeline import Pipeline

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.components.base import AutoSklearnComponent, AutoSklearnChoice
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none


class ColumnTransformer(AutoSklearnComponent):
    def __init__(self, categorical_transformer, numerical_transformer):
        self._branches = [
            ["categorical_branch", categorical_transformer],
            ["numerical_branch", numerical_transformer],
        ]
        self.categorical_features = None

    def _fit(self, X, y=None):
        n_feats = X.shape[1]
        # If categorical_features is none or an array with just False booleans, then
        # just the numerical branch is used
        if self.categorical_features is None or np.all(np.logical_not(self.categorical_features)):
            transformers = [
                [self._branches[1][0], self._branches[1][1], [True] * n_feats]
            ]
        # If all features are categorical, then just just the categorical branch is used 
        elif np.all(self.categorical_features):
            transformers = [
                [self._branches[0][0], self._branches[0][1], [True] * n_feats]
            ]
        # For the other cases, both branches are used
        else:
            transformers = [
                [self._branches[0][0], self._branches[0][1], self.categorical_features],
                [self._branches[1][0], self._branches[1][1], np.logical_not(self.categorical_features)]
            ]
        self.column_transformer = sklearn.compose.ColumnTransformer(transformers)
        return self.column_transformer.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.column_transformer is None:
            raise NotImplementedError()
        return self.column_transformer.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'columntransformer',
                'name': 'Column Transformer',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),}


    def set_hyperparameters(self, configuration, init_params=None):

        if init_params is not None and 'categorical_features' in init_params.keys():
            self.categorical_features = init_params['categorical_features']

        self.configuration = configuration

        for branch_name, branch_transformer in self._branches:

            sub_configuration_space = branch_transformer.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties_
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % branch_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % branch_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space,
                                              values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % branch_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % branch_name, '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(branch_transformer, (AutoSklearnChoice, AutoSklearnComponent, BasePipeline)):
                branch_transformer.set_hyperparameters(configuration=sub_configuration,
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self, dataset_properties=None):
        self.dataset_properties_ = dataset_properties
        cs = ConfigurationSpace()
        cs = ColumnTransformer._get_hyperparameter_search_space_recursevely(
            dataset_properties, cs, self._branches)
        return cs

    @staticmethod
    def _get_hyperparameter_search_space_recursevely(dataset_properties, cs, transformer):
        for st_name, st_operation in transformer:
            if hasattr(st_operation, "get_hyperparameter_search_space"):
                cs.add_configuration_space(st_name,
                    st_operation.get_hyperparameter_search_space(dataset_properties))
            else:
                return ColumnTransformer._get_hyperparameter_search_space_recursevely(
                    dataset_properties, cs, st_operation)
        return cs


