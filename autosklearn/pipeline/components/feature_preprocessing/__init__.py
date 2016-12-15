from collections import OrderedDict
import copy
import os

from autosklearn.pipeline.components.algorithms import AutoSklearnPreprocessingAlgorithm
from ..base import find_components, \
    ThirdPartyComponents
from autosklearn.pipeline.components.choice import ComponentAutoSklearnChoice
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, AbstractConjunction

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(__package__,
                                 classifier_directory,
                                 AutoSklearnPreprocessingAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnPreprocessingAlgorithm)


def add_preprocessor(preprocessor):
    _addons.add_component(preprocessor)


class FeaturePreprocessorChoice(ComponentAutoSklearnChoice):

    def _get_possible_components(self):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    def _get_default_names(self):
        return ['no_preprocessing', 'select_percentile', 'pca', 'truncatedSVD']

    def transform(self, X):
        return self.choice.transform(X)
