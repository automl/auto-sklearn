from typing import Any, Dict, Optional

import os
from collections import OrderedDict

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.base import BaseEstimator

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE

from ...base import (
    AutoSklearnChoice,
    AutoSklearnPreprocessingAlgorithm,
    ThirdPartyComponents,
    _addons,
    find_components,
)

mc_directory = os.path.split(__file__)[0]
_mcs = find_components(__package__, mc_directory, AutoSklearnPreprocessingAlgorithm)
additional_components = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)
_addons["data_preprocessing.minority_coalescense"] = additional_components


def add_mc(mc: BaseEstimator) -> None:
    additional_components.add_component(mc)


class CoalescenseChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls: BaseEstimator) -> Dict[str, BaseEstimator]:
        components: Dict[str, BaseEstimator] = OrderedDict()
        components.update(_mcs)
        components.update(additional_components.components)
        return components

    def get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        default: Optional[str] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No minority coalescers found, please add any one minority coalescer"
                "component."
            )

        if default is None:
            defaults = ["minority_coalescer", "no_coalescense"]
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter(
            "__choice__", list(available_preprocessors.keys()), default_value=default
        )
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[
                name
            ].get_hyperparameter_search_space(dataset_properties=dataset_properties)
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def set_hyperparameters(
        self,
        configuration: Configuration,
        init_params: Optional[Dict[str, Any]] = None,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
    ) -> "CoalescenseChoice":
        new_params = {}

        params = configuration.get_dictionary()
        choice = params["__choice__"]
        del params["__choice__"]

        for param, value in params.items():
            param = param.replace(choice, "").replace(":", "")
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                # These next two lines are different than in the base class -
                # they allow removing the categorical feature indicator array
                #  in order to not pass it to the no encoding
                if choice not in param:
                    continue
                param = param.replace(choice, "").replace(":", "")
                new_params[param] = value

        new_params["random_state"] = self.random_state

        self.new_params = new_params
        new_params["feat_type"] = feat_type
        self.choice = self.get_components()[choice](**new_params)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return self.choice.transform(X)
