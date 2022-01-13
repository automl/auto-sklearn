from collections import OrderedDict
import os

from typing import Any, Dict, Optional

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from sklearn.base import BaseEstimator

from ...base import AutoSklearnPreprocessingAlgorithm, find_components, \
    ThirdPartyComponents, AutoSklearnChoice

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE

bow_directory = os.path.split(__file__)[0]
_bows = find_components(__package__,
                        bow_directory,
                        AutoSklearnPreprocessingAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_bow(bow: 'BagOfWordChoice') -> None:
    _addons.add_component(bow)


class BagOfWordChoice(AutoSklearnChoice):

    @classmethod
    def get_components(cls: BaseEstimator) -> Dict[str, BaseEstimator]:
        components: Dict[str, BaseEstimator] = OrderedDict()
        components.update(_bows)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(
        self,
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
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No bag of word encoders found, please add any bag of word encoder"
                "component.")

        if default is None:
            #ToDo add the different verision
            # how using 'relative' version
            defaults = ['bag_of_words_encoding']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        # Todo how to add hps to available_preprocessors
        preprocessor = CategoricalHyperparameter('__choice__',
                                                 list(
                                                     available_preprocessors.keys()),
                                                 default_value=default)
        #ToDo add hiracical hps for the different choices
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name]. \
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, preprocessor_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def set_hyperparameters(self, configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> 'BagOfWordChoice':
        new_params = {}

        params = configuration.get_dictionary()
        choice = params['__choice__']
        del params['__choice__']

        for param, value in params.items():
            param = param.replace(choice, '').replace(':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                # These next two lines are different than in the base class -
                # they allow removing the categorical feature indicator array
                #  in order to not pass it to the no encoding
                if choice not in param:
                    continue
                param = param.replace(choice, '').replace(':', '')
                new_params[param] = value

        new_params['random_state'] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return self.choice.transform(X)