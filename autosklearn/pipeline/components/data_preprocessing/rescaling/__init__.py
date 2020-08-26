from collections import OrderedDict
import os
from ...base import AutoSklearnPreprocessingAlgorithm, find_components, \
    ThirdPartyComponents, AutoSklearnChoice
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


rescaling_directory = os.path.split(__file__)[0]
_rescalers = find_components(__package__,
                             rescaling_directory,
                             AutoSklearnPreprocessingAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_rescaler(rescaler):
    _addons.add_component(rescaler)


class RescalingChoice(AutoSklearnChoice):

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_rescalers)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(self, dataset_properties=None,
                                        default=None,
                                        include=None,
                                        exclude=None):
        cs = ConfigurationSpace()

        if dataset_properties is None:
            dataset_properties = {}

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No rescalers found, please add any rescaling component.")

        if default is None:
            defaults = ['standardize', 'none', 'minmax', 'normalize']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter('__choice__',
                                                 list(
                                                     available_preprocessors.keys()),
                                                 default_value=default)
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

    def transform(self, X):
        return self.choice.transform(X)
