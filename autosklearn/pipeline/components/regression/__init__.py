from collections import OrderedDict
import copy
import os

from ..base import AutoSklearnRegressionAlgorithm, find_components, \
    ThirdPartyComponents, AutoSklearnChoice
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

regressor_directory = os.path.split(__file__)[0]
_regressors = find_components(__package__,
                              regressor_directory,
                              AutoSklearnRegressionAlgorithm)
_addons = ThirdPartyComponents(AutoSklearnRegressionAlgorithm)


def add_regressor(regressor):
    _addons.add_component(regressor)


class RegressorChoice(AutoSklearnChoice):

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_regressors)
        components.update(_addons.components)
        return components

    @classmethod
    def get_available_components(cls, data_prop,
                                 include=None,
                                 exclude=None):
        available_comp = cls.get_components()
        components_dict = OrderedDict()

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

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
            if entry == RegressorChoice:
                continue

            if entry.get_properties()['handles_regression'] is False:
                continue
            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(self, dataset_properties,
                                        default=None,
                                        include=None,
                                        exclude=None):
        if include is not None and exclude is not None:
            raise ValueError("The argument include and exclude cannot be used together.")

        cs = ConfigurationSpace()

        # Compile a list of all estimator objects for this problem
        available_estimators = self.get_available_components(
            data_prop=dataset_properties,
            include=include,
            exclude=exclude)

        if len(available_estimators) == 0:
            raise ValueError("No regressors found")

        if default is None:
            defaults = ['random_forest', 'support_vector_regression'] + \
                list(available_estimators.keys())
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
            parent_hyperparameter = {'parent': estimator, 'value': estimator_name}
            cs.add_configuration_space(estimator_name, estimator_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space_ = cs
        self.dataset_properties_ = dataset_properties
        return cs

    def estimator_supports_iterative_fit(self):
        return hasattr(self.choice, 'iterative_fit')

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        if fit_params is None:
            fit_params = {}
        return self.choice.iterative_fit(X, y, n_iter=n_iter, **fit_params)

    def configuration_fully_fitted(self):
        return self.choice.configuration_fully_fitted()
