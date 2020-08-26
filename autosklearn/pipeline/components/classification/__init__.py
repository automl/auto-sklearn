__author__ = 'feurerm'

from collections import OrderedDict
import os

from ..base import AutoSklearnClassificationAlgorithm, find_components, \
    ThirdPartyComponents, AutoSklearnChoice
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__,
                               classifier_directory,
                               AutoSklearnClassificationAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnClassificationAlgorithm)


def add_classifier(classifier):
    _addons.add_component(classifier)


class ClassifierChoice(AutoSklearnChoice):

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_classifiers)
        components.update(_addons.components)
        return components

    def get_available_components(cls, dataset_properties=None,
                                 include=None,
                                 exclude=None):
        if dataset_properties is None:
            dataset_properties = {}

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
            if dataset_properties.get('multiclass') is True and \
               entry.get_properties()['handles_multiclass'] is False:
                continue
            if dataset_properties.get('multilabel') is True and \
               available_comp[name].get_properties()['handles_multilabel'] is False:
                continue
            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(self, dataset_properties=None,
                                        default=None,
                                        include=None,
                                        exclude=None):
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_estimators cannot be used together.")

        cs = ConfigurationSpace()

        # Compile a list of all estimator objects for this problem
        available_estimators = self.get_available_components(
            dataset_properties=dataset_properties,
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
                                              default_value=default)
        cs.add_hyperparameter(estimator)
        for estimator_name in available_estimators.keys():
            estimator_configuration_space = available_estimators[estimator_name].\
                get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': estimator,
                                     'value': estimator_name}
            cs.add_configuration_space(estimator_name,
                                       estimator_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def predict_proba(self, X):
        return self.choice.predict_proba(X)

    def estimator_supports_iterative_fit(self):
        return hasattr(self.choice, 'iterative_fit')

    def get_max_iter(self):
        if self.estimator_supports_iterative_fit():
            return self.choice.get_max_iter()
        else:
            raise NotImplementedError()

    def get_current_iter(self):
        if self.estimator_supports_iterative_fit():
            return self.choice.get_current_iter()
        else:
            raise NotImplementedError()

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        # Allows to use check_is_fitted on the choice object
        self.fitted_ = True
        if fit_params is None:
            fit_params = {}
        return self.choice.iterative_fit(X, y, n_iter=n_iter, **fit_params)

    def configuration_fully_fitted(self):
        return self.choice.configuration_fully_fitted()
