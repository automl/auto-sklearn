__author__ = 'feurerm'

from collections import OrderedDict
import copy
import os

from ..base import AutoSklearnClassificationAlgorithm, find_components, \
    ThirdPartyComponents
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__,
                               classifier_directory,
                               AutoSklearnClassificationAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnClassificationAlgorithm)


def add_classifier(classifier):
    _addons.add_component(classifier)


class ClassifierChoice(object):
    def __init__(self, **params):
        choice = params['__choice__']
        del params['__choice__']
        self.choice = self.get_components()[choice](**params)

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_classifiers)
        components.update(_addons.components)
        return components

    @classmethod
    def get_available_components(cls, data_prop,
                                 include=None,
                                 exclude=None):
        available_comp = cls.get_components()
        components_dict = OrderedDict()

        if include is not None and exclude is not None:
            raise ValueError("The argument include and exclude cannot be used together.")

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Avoid infinite loop
            if entry == ClassifierChoice:
                continue

            if entry.get_properties()['handles_classification'] is False:
                continue
            if data_prop.get('multiclass') is True and entry.get_properties()[
                'handles_multiclass'] is False:
                continue
            if data_prop.get('multilabel') is True and available_comp[name]. \
                    get_properties()['handles_multilabel'] is False:
                continue
            components_dict[name] = entry

        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, dataset_properties,
                                        default=None,
                                        include=None,
                                        exclude=None):
        if include is not None and exclude is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_estimators cannot be used together.")

        cs = ConfigurationSpace()

        # Compile a list of all estimator objects for this problem
        available_estimators = cls.get_available_components(
            data_prop=dataset_properties,
            include=include,
            exclude=exclude)

        if len(available_estimators) == 0:
            raise ValueError("No classifiers found")

        if default is None:
            defaults = ['random_forest', 'liblinear_svc', 'sgd',
                        'libsvm_svc'] + list(available_estimators.keys())
            for default_ in defaults:
                if default_ in available_estimators:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        estimator = CategoricalHyperparameter('__choice__',
                                              list(available_estimators.keys()),
                                              default=default)
        cs.add_hyperparameter(estimator)
        for estimator_name in available_estimators.keys():
            # We have to retrieve the configuration space every time because
            # we change the objects it returns. If we reused it, we could not
            # retrieve the conditions further down
            # TODO implement copy for hyperparameters and forbidden and
            # conditions!

            estimator_configuration_space = available_estimators[
                estimator_name]. \
                get_hyperparameter_search_space(dataset_properties)
            for parameter in estimator_configuration_space.get_hyperparameters():
                new_parameter = copy.deepcopy(parameter)
                new_parameter.name = "%s:%s" % (
                    estimator_name, new_parameter.name)
                cs.add_hyperparameter(new_parameter)
                # We must only add a condition if the hyperparameter is not
                # conditional on something else
                if len(estimator_configuration_space.
                        get_parents_of(parameter)) == 0:
                    condition = EqualsCondition(new_parameter, estimator,
                                                estimator_name)
                    cs.add_condition(condition)

            for condition in available_estimators[estimator_name]. \
                    get_hyperparameter_search_space(
                    dataset_properties).get_conditions():
                dlcs = condition.get_descendant_literal_conditions()
                for dlc in dlcs:
                    if not dlc.child.name.startswith(estimator_name):
                        dlc.child.name = "%s:%s" % (
                            estimator_name, dlc.child.name)
                    if not dlc.parent.name.startswith(estimator_name):
                        dlc.parent.name = "%s:%s" % (
                            estimator_name, dlc.parent.name)
                cs.add_condition(condition)

            for forbidden_clause in available_estimators[estimator_name]. \
                    get_hyperparameter_search_space(
                    dataset_properties).forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(estimator_name):
                        dlc.hyperparameter.name = "%s:%s" % (estimator_name,
                                                             dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)
    
        return cs
