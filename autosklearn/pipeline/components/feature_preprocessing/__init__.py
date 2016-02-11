from collections import OrderedDict
import copy
import importlib
import inspect
import os
import pkgutil
import sys

from ..base import AutoSklearnPreprocessingAlgorithm, find_components, \
    ThirdPartyComponents
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


class FeaturePreprocessorChoice(object):
    def __init__(self, **params):
        choice = params['__choice__']
        del params['__choice__']
        self.choice = self.get_components()[choice](**params)

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        components.update(_addons.components)
        return components

    @classmethod
    def get_available_components(cls, data_prop,
                                 include=None,
                                 exclude=None):
        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = cls.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        # TODO check for task type classification and/or regression!

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Exclude itself to avoid infinite loop
            if entry == FeaturePreprocessorChoice or hasattr(entry, 'get_components'):
                continue

            target_type = data_prop['target_type']
            if target_type == 'classification':
                if entry.get_properties()['handles_classification'] is False:
                    continue
                if data_prop.get('multiclass') is True and \
                        entry.get_properties()['handles_multiclass'] is False:
                    continue
                if data_prop.get('multilabel') is True and \
                        entry.get_properties()['handles_multilabel'] is False:
                    continue

            elif target_type == 'regression':
                if entry.get_properties()['handles_regression'] is False:
                    continue

            else:
                raise ValueError('Unknown target type %s' % target_type)

            components_dict[name] = entry

        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, dataset_properties,
                                        default=None,
                                        include=None,
                                        exclude=None):
        cs = ConfigurationSpace()

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = cls.get_available_components(
            data_prop=dataset_properties,
            include=include, exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError(
                "No preprocessors found, please add NoPreprocessing")

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
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name]. \
                get_hyperparameter_search_space(dataset_properties)
            for parameter in preprocessor_configuration_space.get_hyperparameters():
                new_parameter = copy.deepcopy(parameter)
                new_parameter.name = "%s:%s" % (name, new_parameter.name)
                cs.add_hyperparameter(new_parameter)
                # We must only add a condition if the hyperparameter is not
                # conditional on something else
                if len(preprocessor_configuration_space.
                        get_parents_of(parameter)) == 0:
                    condition = EqualsCondition(new_parameter, preprocessor,
                                                name)
                    cs.add_condition(condition)

            for condition in available_preprocessors[name]. \
                    get_hyperparameter_search_space(
                    dataset_properties).get_conditions():
                if not isinstance(condition, AbstractConjunction):
                    dlcs = [condition]
                else:
                    dlcs = condition.get_descendent_literal_conditions()
                for dlc in dlcs:
                    if not dlc.child.name.startswith(name):
                        dlc.child.name = "%s:%s" % (name, dlc.child.name)
                    if not dlc.parent.name.startswith(name):
                        dlc.parent.name = "%s:%s" % (name, dlc.parent.name)
                cs.add_condition(condition)

            for forbidden_clause in available_preprocessors[name]. \
                    get_hyperparameter_search_space(
                    dataset_properties).forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                                                             dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)

        return cs
