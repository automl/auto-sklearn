from autosklearn.pipeline.components.algorithms import AutoSklearnClassificationAlgorithm

__author__ = 'feurerm'

from collections import OrderedDict
import copy
import os

from ..base import find_components, \
    ThirdPartyComponents
from autosklearn.pipeline.components.choice import ComponentAutoSklearnChoice, ClassificationComponentFilter
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__,
                               classifier_directory,
                               AutoSklearnClassificationAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnClassificationAlgorithm)


def add_classifier(classifier):
    _addons.add_component(classifier)


class ClassifierChoice(ComponentAutoSklearnChoice):

    def _get_possible_components(cls):
        components = OrderedDict()
        components.update(_classifiers)
        components.update(_addons.components)
        return components

    def _get_default_names(self):
        return ['random_forest', 'liblinear_svc', 'sgd', 'libsvm_svc'] \
                   + list(self.available_components.keys())

    def predict_proba(self, X):
        return self.choice.predict_proba(X)
