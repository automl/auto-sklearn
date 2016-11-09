from autosklearn.pipeline.components.composite import CompositeAutoSklearnComponent, CompositeConfigSpaceBuilder


class ParallelAutoSklearnComponent(CompositeAutoSklearnComponent):

    def get_config_space_builder(self):
        builder = ParallelConfigSpaceBuilder(self)
        for name, component in self.components.items():
            child = component.get_config_space_builder()
            builder.add_child(name, child)
        return builder


class ParallelConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def explore_data_flow(self, data_description):
        data_descriptions = []
        for name, node in self._children.items():
            data_description_copy = data_description.copy()
            current_step_data_descriptions = node.explore_data_flow(data_description_copy)
            data_descriptions.extend(current_step_data_descriptions)
        return data_descriptions