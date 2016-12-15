from collections import OrderedDict
import copy
import os
from autosklearn.pipeline.components.algorithms import AutoSklearnPreprocessingAlgorithm
from ...base import find_components, \
    ThirdPartyComponents
from autosklearn.pipeline.components.choice import ComponentAutoSklearnChoice
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter


rescaling_directory = os.path.split(__file__)[0]
_rescalers = find_components(__package__,
                             rescaling_directory,
                             AutoSklearnPreprocessingAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_rescaler(rescaler):
    _addons.add_component(rescaler)


class RescalingChoice(ComponentAutoSklearnChoice):

    def _get_possible_components(self):
        components = OrderedDict()
        components.update(_rescalers)
        components.update(_addons.components)
        return components

    def _get_default_names(self):
        return ['standardize', 'none', 'minmax', 'normalize']

    def transform(self, X):
        return self.choice.transform(X)