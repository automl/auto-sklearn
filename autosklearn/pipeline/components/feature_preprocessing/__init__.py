from collections import OrderedDict
import copy
import os

from autosklearn.pipeline.components.algorithms import AutoSklearnPreprocessingAlgorithm
from ..base import find_components, \
    ThirdPartyComponents
from autosklearn.pipeline.components.choice import AutoSklearnChoice
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


class FeaturePreprocessorChoice(AutoSklearnChoice):

    def __init__(self,
                 target_type='all',
                 is_multiclass=False,
                 is_multilabel=False,
                 random_state=None,
                 include=None,
                 exclude=None,
                 default=None):
        self.target_type = target_type
        self.is_multiclass = is_multiclass
        self.is_multilabel = is_multilabel
        super(FeaturePreprocessorChoice, self).__init__(include, exclude, default, random_state)

    def get_components(self):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    def get_available_components(self):
        '''
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)
        '''

        # TODO check for task type classification and/or regression!

        available_comp = self.get_components()
        components_dict = OrderedDict()
        for name in available_comp:
            if self.include is not None and name not in self.include:
                continue
            elif self.exclude is not None and name in self.exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == FeaturePreprocessorChoice or hasattr(entry, 'get_components'):
                continue

            target_type = self.target_type
            if target_type == 'classification':
                if entry.get_properties()['handles_classification'] is False:
                    continue
                if self.is_multiclass and \
                        entry.get_properties()['handles_multiclass'] is False:
                    continue
                if self.is_multilabel is True and \
                        entry.get_properties()['handles_multilabel'] is False:
                    continue

            elif target_type == 'regression':
                if entry.get_properties()['handles_regression'] is False:
                    continue

            else:
                raise ValueError('Unknown target type %s' % target_type)

            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components()

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No preprocessors found, please add NoPreprocessing")

        default = self.default
        if default is None:
            defaults = ['no_preprocessing', 'select_percentile', 'pca',
                        'truncatedSVD']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter('__choice__',
                                                 list(
                                                     available_preprocessors.keys()),
                                                 default=default)
        cs.add_hyperparameter(preprocessor)
        return cs

    def transform(self, X):
        return self.choice.transform(X)
