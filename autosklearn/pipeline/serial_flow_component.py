from collections import OrderedDict

from autosklearn.pipeline.components.base import AutoSklearnComponent
from autosklearn.pipeline.graph_based_config_space import ParallelConfigSpaceBuilder, \
    SerialConfigSpaceBuilder


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


class SerialFlow(CompositeAutoSklearnComponent):

    def get_config_space_builder(self):
        builder = SerialConfigSpaceBuilder(self)
        for name, component in self.components.items():
            child = component.get_config_space_builder()
            builder.add_child(name, child)
        return builder


class ParallelFlow(CompositeAutoSklearnComponent):

    def get_config_space_builder(self):
        builder = ParallelConfigSpaceBuilder(self)
        for name, component in self.components.items():
            child = component.get_config_space_builder()
            builder.add_child(name, child)
        return builder