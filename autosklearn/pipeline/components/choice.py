from collections import OrderedDict

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter
from sklearn.utils import check_random_state

from autosklearn.pipeline.components.composite import CompositeAutoSklearnComponent, CompositeConfigSpaceBuilder


class AutoSklearnChoice(CompositeAutoSklearnComponent):

    def __init__(self, include=None, exclude=None, default=None, random_state=None):
        self.include = include
        self.exclude = exclude

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        components = self.get_available_components()
        components = [(name, cls()) for (name, cls) in components.items()]
        super(AutoSklearnChoice, self).__init__(components)

        if default:
            self.default = default
            self.choice = self.components[default]
        else:
            default = self._get_default_name()
            self.default = default
            self.choice = self.components[default]

    def _get_default_name(self):
        for default_name, default_obj in self.components.items():
            return default_name

    def get_components(cls):
        raise NotImplementedError()

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
        available_comp = self.get_components()
        components_dict = OrderedDict()
        for name in available_comp:
            if self.include is not None and name not in self.include:
                continue
            elif self.exclude is not None and name in self.exclude:
                continue

            # TODO maybe check for sparse?

            components_dict[name] = available_comp[name]

        return components_dict

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()

        options = list(self.components.keys())
        choice = CategoricalHyperparameter('__choice__', options,
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
