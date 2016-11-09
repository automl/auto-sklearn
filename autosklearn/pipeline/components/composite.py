from collections import OrderedDict

from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace

from autosklearn.pipeline.components.base import AutoSklearnComponent
from autosklearn.pipeline.components.config_space import ConfigSpaceBuilder


class CompositeAutoSklearnComponent(AutoSklearnComponent):

    def __init__(self, components):
        self.components = OrderedDict()
        for component in components:
            if isinstance(component, AutoSklearnComponent):
                self._add_component(component)
                continue
            elif isinstance(component, tuple):
                name, component = component
                if isinstance(name, str) and isinstance(component, AutoSklearnComponent):
                    self.components[name] = component
                    continue
            raise ValueError(component)

    def _add_component(self, component, name=None):
        if not name:
            name = type(component).__name__.lower()
        temp_name = name
        counter = 1
        while temp_name in self.components:
            counter += 1
            temp_name = name + str(counter)

        self.components[temp_name] = component

    def get_config_space(self):
        builder = self.get_config_space_builder()
        cs = builder.build()
        return cs

    def get_config_space_builder(self):
        pass

    def set_hyperparameters(self, configuration):
        if isinstance(configuration, Configuration):
            configuration = configuration.get_dictionary()

        component_configs = {}
        for component_name in self.components:
            component_configs[component_name] = {}
        for hp_name, hp_value in configuration.items():
            separator_index = hp_name.index(':')
            component_name = hp_name[:separator_index]
            sub_hp_name = hp_name[separator_index+1:]
            component_configs[component_name][sub_hp_name] = hp_value
        for component_name, config in component_configs.items():
            component = self.components[component_name]
            component.set_hyperparameters(config)


class CompositeConfigSpaceBuilder(ConfigSpaceBuilder):

    def __init__(self, element):
        super(CompositeConfigSpaceBuilder, self).__init__(element)
        self._children = OrderedDict()

    def add_child(self, name, node):
        node._set_parent(self)
        node._set_name(name)
        self._children[name] = node

    def get_config_space(self):
        cs = ConfigurationSpace()
        for name, node in self._children.items():
            sub_cs = node.get_config_space()
            cs.add_configuration_space(name, sub_cs)
        return cs
