from collections import OrderedDict

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter
from sklearn.utils import check_random_state

from autosklearn.pipeline.components.composite import CompositeAutoSklearnComponent, CompositeConfigSpaceBuilder


class AutoSklearnChoice(CompositeAutoSklearnComponent):

    def __init__(self, components, default):
        super(AutoSklearnChoice, self).__init__(components)
        self.default = default
        self.choice = self.components[default]

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()

        options = list(self.components.keys())
        choice = CategoricalHyperparameter('__choice__',
                                           choices=options,
                                           default=self.default)
        cs.add_hyperparameter(choice)

        return cs

    def set_hyperparameters(self, configuration):
        if isinstance(configuration, Configuration):
            configuration = configuration.get_dictionary()

        choice = configuration['__choice__']
        del configuration['__choice__']
        self.choice = self.components[choice]

        component_config = {}
        for hp_name, hp_value in configuration.items():
            separator_index = hp_name.index(':')
            component_name = hp_name[:separator_index]
            if component_name == choice:
                sub_hp_name = hp_name[separator_index+1:]
                component_config[sub_hp_name] = hp_value
        self.choice.set_hyperparameters(component_config)

    def fit(self, X, y, **kwargs):
        if kwargs is None:
            kwargs = {}
        return self.choice.fit(X, y, **kwargs)

    def predict(self, X):
        return self.choice.predict(X)

    def get_config_space_builder(self):
        builder = ChoiceConfigSpaceBuilder(self)
        for name, component in self.components.items():
            child = component.get_config_space_builder()
            builder.add_child(name, child)
        return builder


class ChoiceConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def get_config_space(self):
        cs = self._element.get_hyperparameter_search_space()
        choice_parameter = cs.get_hyperparameter('__choice__')
        for name, node in self._children.items():
            sub_cs = node.get_config_space()
            cs.add_configuration_space(name, sub_cs, parent_hyperparameter={'parent': choice_parameter, 'value': name})
        return cs

    def explore_data_flow(self, data_description):
        data_descriptions = []
        for name, node in self._children.items():
            data_description_copy = data_description.copy()
            data_description_copy.add_choice(self, name)
            current_step_data_descriptions = node.explore_data_flow(data_description_copy)
            data_descriptions.extend(current_step_data_descriptions)
        return data_descriptions


class ComponentAutoSklearnChoice(AutoSklearnChoice):

    def __init__(self, filter=None, default=None):
        self.filter = filter if filter else ComponentFilter()

        self.available_components = self._get_available_components()
        if not default:
            default = self._get_default_name()

        components = [(name, cls()) for (name, cls) in self.available_components.items()]
        super(ComponentAutoSklearnChoice, self).__init__(components, default)

    def _get_default_name(self):
        for default in self._get_default_names():
            if default in self.available_components:
                return default

    def _get_default_names(self):
        return self.components.keys()

    def _get_possible_components(cls):
        raise NotImplementedError()

    def _get_available_components(self):
        possible_components = self._get_possible_components()
        available_components = OrderedDict()
        for name, component in possible_components.items():
            if self.filter.can_be_used(name, component):
                available_components[name] = possible_components[name]
        return available_components


class ComponentFilter():

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def can_be_used(self, name, component):
        return self._can_be_included(name) and self._can_be_used(component)

    def _can_be_included(self, name):
        if self.include is not None and name not in self.include:
            return False
        elif self.exclude is not None and name in self.exclude:
            return False
        return True

    def _can_be_used(self, component):
        return True


class ClassificationComponentFilter(ComponentFilter):

    def __init__(self,
                 include=None,
                 exclude=None,
                 is_multiclass=False,
                 is_multilabel=False):
        super(ClassificationComponentFilter, self).__init__(include, exclude)
        self.is_multiclass = is_multiclass
        self.is_multilabel = is_multilabel

    def _can_be_used(self, component):
        if component.get_properties()['handles_classification'] is False:
            return False
        if self.is_multiclass and \
                        component.get_properties()['handles_multiclass'] is False:
            return False
        if self.is_multilabel is True and \
                        component.get_properties()['handles_multilabel'] is False:
            return False
        return True


class RegressionComponentFilter(ComponentFilter):

    def _can_be_used(self, component):
        return component.get_properties()['handles_regression']
