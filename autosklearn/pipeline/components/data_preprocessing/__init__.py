from typing import Dict, Optional, Type

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import PIPELINE_DATA_DTYPE

from ..base import (
    AutoSklearnChoice,
    AutoSklearnPreprocessingAlgorithm,
    ThirdPartyComponents,
    find_components,
)

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(
    __package__, classifier_directory, AutoSklearnPreprocessingAlgorithm
)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_preprocessor(preprocessor: Type[AutoSklearnPreprocessingAlgorithm]) -> None:
    _addons.add_component(preprocessor)


class DataPreprocessorChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls) -> OrderedDict:
        components: OrderedDict = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    def get_available_components(
        self,
        dataset_properties: Optional[Dict] = None,
        include: Optional[Dict] = None,
        exclude: Optional[Dict] = None,
    ) -> OrderedDict:
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together."
            )

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError(
                        "Trying to include unknown component: " "%s" % incl
                    )

        # TODO check for task type classification and/or regression!

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == DataPreprocessorChoice or hasattr(entry, "get_components"):
                continue

            target_type = dataset_properties["target_type"]
            if target_type == "classification":
                if entry.get_properties()["handles_classification"] is False:
                    continue
                if (
                    dataset_properties.get("multiclass") is True
                    and entry.get_properties()["handles_multiclass"] is False
                ):
                    continue
                if (
                    dataset_properties.get("multilabel") is True
                    and entry.get_properties()["handles_multilabel"] is False
                ):
                    continue

            elif target_type == "regression":
                if entry.get_properties()["handles_regression"] is False:
                    continue
                if (
                    dataset_properties.get("multioutput") is True
                    and entry.get_properties()["handles_multioutput"] is False
                ):
                    continue

            else:
                raise ValueError("Unknown target type %s" % target_type)

            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[Dict] = None,
        default: str = None,
        include: Optional[Dict] = None,
        exclude: Optional[Dict] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties, include=include, exclude=exclude
        )

        if len(available_preprocessors) == 0:
            raise ValueError("No preprocessors found, please add NoPreprocessing")

        if default is None:
            defaults = ["feature_type"]
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter(
            "__choice__", list(available_preprocessors.keys()), default_value=default
        )
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name](
                feat_type=feat_type, dataset_properties=dataset_properties
            ).get_hyperparameter_search_space(dataset_properties=dataset_properties)
            parent_hyperparameter = {"parent": preprocessor, "value": name}
            cs.add_configuration_space(
                name,
                preprocessor_configuration_space,
                parent_hyperparameter=parent_hyperparameter,
            )
        return cs

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return self.choice.transform(X)

    def set_hyperparameters(
        self,
        configuration: ConfigurationSpace,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        init_params: Optional[Dict] = None,
    ) -> "DataPreprocessorChoice":
        config = {}
        params = configuration.get_dictionary()
        choice = params["__choice__"]
        del params["__choice__"]

        for param, value in params.items():
            param = param.replace(choice, "").split(":", 1)[1]
            config[param] = value

        new_params = {}
        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, "").split(":", 1)[-1]
                if "feat_type" in param:
                    feat_type = value
                else:
                    new_params[param] = value
        self.choice = self.get_components()[choice](
            config=config, init_params=new_params, feat_type=feat_type
        )

        return self
