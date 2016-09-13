from collections import defaultdict, OrderedDict

from ConfigSpace import ConfigurationSpace


class InvalidDataArtifactsException(Exception):

    def __init__(self, artifacts):
        self.artifacts = artifacts
        message = "Unable to transform data with: %s" % artifacts
        super(InvalidDataArtifactsException, self).__init__(message)

class ConfigSpaceBuilder(object):

    def get_config_space(self):
        pass

    def explore_data_flow(self, data_description):
        pass

    def get_incompatible_nodes(self, cs):
        data_description = DataDescription()
        data_descriptions = self.explore_data_flow(data_description)
        incompatible_nodes = [x for x in data_descriptions if isinstance(x, IncompatibleNodes)]

        return incompatible_nodes

    def build(self):
        cs = self.get_config_space()
        self.get_incompatible_nodes(cs)

        return cs


class LeafNodeConfigSpaceBuilder(ConfigSpaceBuilder):

    def __init__(self, element):
        self._element = element
        self._parent = None
        self._children = OrderedDict()

    def _set_name(self, name):
        self._name = name

    def _set_parent(self, parent):
        self._parent = parent

    def add_child(self, name, node):
        node._set_parent(self)
        node._set_name(name)
        self._children[name] = node

    def get_config_space(self):
        return self._element.get_config_space()

    def explore_data_flow(self, data_description):
        try:
            artifacts = data_description.get_artifacts()
            artifacts = self._element.transform_data_description(artifacts)
            data_description.update_artifacts(self, artifacts)
            return [data_description]
        except InvalidDataArtifactsException as ex:
            for artifact in ex.artifacts:
                nodes = []
                source_node = data_description.get_node_by_artifact(artifact)
                choices = data_description.get_choices()
                nodes.append(IncompatibleNodes(artifact, source_node, self, choices))
                return nodes


class CompositeConfigSpaceBuilder(LeafNodeConfigSpaceBuilder):

    def get_config_space(self):
        cs = ConfigurationSpace()
        for name, node in self._children.items():
            sub_cs = node.get_config_space()
            cs.add_configuration_space(name, sub_cs)
        return cs


class ParallelConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def explore_data_flow(self, data_description):
        data_descriptions = []
        for name, node in self._children.items():
            data_description_copy = data_description.copy()
            data_description_copy.add_choice(self, name)
            current_step_data_descriptions = node.explore_data_flow(data_description_copy)
            data_descriptions.extend(current_step_data_descriptions)
        return data_descriptions


class SerialConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def explore_data_flow(self, data_description):
        data_descriptions = [data_description]
        for name, node in self._children.items():
            current_step_data_descriptions = []
            for data_description in data_descriptions:
                if isinstance(data_description, DataDescription):
                    data_descriptions = node.explore_data_flow(data_description)
                    current_step_data_descriptions.extend(data_descriptions)
                else:
                    current_step_data_descriptions.append(data_description)
            data_descriptions = current_step_data_descriptions
        return data_descriptions


class ChoiceConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def explore_data_flow(self, data_description):
        data_descriptions = []
        for name, node in self._children.items():
            data_description_copy = data_description.copy()
            current_step_data_descriptions = node.explore_data_flow(data_description_copy)
            data_descriptions.extend(current_step_data_descriptions)
        return data_descriptions


class DataDescription(object):

    def __init__(self):
        self._node_by_artifact = {}
        self._choices = []

    def copy(self):
        data_description = DataDescription()
        nodes_by_artifacts = self._node_by_artifact.copy()
        data_description._node_by_artifact = nodes_by_artifacts
        return data_description

    def get_artifacts(self):
        return list(self._node_by_artifact.keys())

    def update_artifacts(self, node, artifacts):
        current_artifacts = set(self._node_by_artifact.keys())
        updated_artifacts = set(artifacts)

        artifacts_to_add = updated_artifacts.difference(current_artifacts)
        for artifact in artifacts_to_add:
            self.add_artifact(node, artifact)

        artifacts_to_remove = current_artifacts.difference(artifacts_to_add)
        for artifact in artifacts_to_remove:
            self.remove_artifact(artifact)

    def add_artifact(self, node, artifact):
        self._node_by_artifact[artifact] = node

    def remove_artifact(self, artifact):
        if artifact in self._node_by_artifact:
            del self._node_by_artifact[artifact]

    def get_node_by_artifact(self, artifact):
        return self._node_by_artifact[artifact]

    def add_choice(self, node, option):
        self._choices.append((node, option))

    def get_choices(self):
        return self._choices


class IncompatibleNodes(object):

    def __init__(self, artifact, source_node, final_node, choices):
        self.artifact = artifact
        self.source_node = source_node
        self.final_node = final_node
        self.choices = choices
