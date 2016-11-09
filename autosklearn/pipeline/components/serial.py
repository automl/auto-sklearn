from autosklearn.pipeline.components.composite import CompositeAutoSklearnComponent, CompositeConfigSpaceBuilder


class SerialAutoSklearnComponent(CompositeAutoSklearnComponent):

    def get_config_space_builder(self):
        builder = SerialConfigSpaceBuilder(self)
        for name, component in self.components.items():
            child = component.get_config_space_builder()
            builder.add_child(name, child)
        return builder


class SerialConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def explore_data_flow(self, data_description):
        data_descriptions = [data_description]
        for name, node in self._children.items():
            current_step_data_descriptions = []
            for data_description in data_descriptions:
                if data_description.is_valid:
                    data_descriptions = node.explore_data_flow(data_description)
                    current_step_data_descriptions.extend(data_descriptions)
                else:
                    current_step_data_descriptions.append(data_description)
            data_descriptions = current_step_data_descriptions
        return data_descriptions