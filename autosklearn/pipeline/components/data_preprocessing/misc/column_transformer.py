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
    def __init__(self, transformers):
        """[summary]
        
        Parameters
        ----------
        transformers : [list]
            List of (name, transformer, column(s)) tuples specifying the transformer 
            objects to be applied to subsets of the data.
        """
        self.transformers = transformers
        

    def _fit(self, X, y=None):
        self.column_transformer = sklearn.compose.ColumnTransformer(self.transformers)
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

    #def set_hyperparameters(self, configuration, init_params=None):
    #    for transf in self.transformers:
    #        #transf_name, transf_operation = transf[0], transf[1]
    #        transf[1].set_hyperparameters(configuration, init_params)
    #    return self


    def set_hyperparameters(self, configuration, init_params=None):
        self.configuration = configuration

        for transf in self.transformers:
            transf_name, transf_operation = transf[0], transf[1]

            sub_configuration_space = transf_operation.get_hyperparameter_search_space(
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

            if isinstance(transf_operation, (AutoSklearnChoice, AutoSklearnComponent, BasePipeline)):
                transf_operation.set_hyperparameters(configuration=sub_configuration,
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self


    def get_hyperparameter_search_space(self, dataset_properties=None):
        self.dataset_properties_ = dataset_properties
        cs = ConfigurationSpace()
        cs = ColumnTransformer._get_hyperparameter_search_space_recursevely(
            dataset_properties, cs, self.transformers)
        return cs

    @staticmethod
    def _get_hyperparameter_search_space_recursevely(dataset_properties, cs, transformer):
        for sub_transf in transformer:
            st_name, st_operation = sub_transf[0], sub_transf[1]
            if hasattr(st_operation, "get_hyperparameter_search_space"):
                cs.add_configuration_space(st_name,
                    st_operation.get_hyperparameter_search_space(dataset_properties))
            else:
                return ColumnTransformer._get_hyperparameter_search_space_recursevely(
                    dataset_properties, cs, st_operation)
        return cs


