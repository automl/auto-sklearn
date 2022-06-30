from typing import Dict, Optional, Type, Union

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.askl_typing import EXCLUDE_BASE_TYPE, FEAT_TYPE_TYPE, INCLUDE_BASE_TYPE

from ..base import (
    AutoSklearnChoice,
    AutoSklearnPreprocessingAlgorithm,
    ThirdPartyComponents,
    _addons,
    find_components,
)

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(
    __package__, classifier_directory, AutoSklearnPreprocessingAlgorithm
)
additional_components = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)
_addons["feature_preprocessing"] = additional_components

DATASET_PROPERTIES_TYPE = Dict[str, Union[str, int, bool]]


def add_preprocessor(preprocessor: Type[AutoSklearnPreprocessingAlgorithm]) -> None:
    additional_components.add_component(preprocessor)


class FeaturePreprocessorChoice(AutoSklearnChoice):
    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(additional_components.components)
        return components

    def get_available_components(
        self,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        include: Optional[INCLUDE_BASE_TYPE] = None,
        exclude: Optional[EXCLUDE_BASE_TYPE] = None,
    ):
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
            if entry == FeaturePreprocessorChoice or hasattr(entry, "get_components"):
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
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        default=None,
        include: Optional[INCLUDE_BASE_TYPE] = None,
        exclude: Optional[EXCLUDE_BASE_TYPE] = None,
    ):
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
            defaults = ["no_preprocessing", "select_percentile", "pca", "truncatedSVD"]
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

        return cs

    def transform(self, X):
        return self.choice.transform(X)
