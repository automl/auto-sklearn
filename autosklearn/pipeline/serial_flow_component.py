from collections import OrderedDict

from autosklearn.pipeline.components.base import AutoSklearnComponent, CompositeAutoSklearnComponent
from autosklearn.pipeline.graph_based_config_space import ParallelConfigSpaceBuilder, \
    SerialConfigSpaceBuilder




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