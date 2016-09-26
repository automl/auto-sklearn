from collections import defaultdict, OrderedDict
from copy import copy

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
        incompatible_nodes = [x for x in data_descriptions if x.is_incompatible]

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
        return self._element.get_hyperparameter_search_space()

    def explore_data_flow(self, data_description):
        try:
            artifacts = data_description.get_artifacts()
            artifacts = self._element.transform_data_description(artifacts)
            data_description.update_artifacts(self, artifacts)
            return [data_description]
        except InvalidDataArtifactsException as ex:
            return [IncompatibleDataDescription(data_description, self, ex.artifacts)]


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
                if data_description.is_valid:
                    data_descriptions = node.explore_data_flow(data_description)
                    current_step_data_descriptions.extend(data_descriptions)
                else:
                    current_step_data_descriptions.append(data_description)
            data_descriptions = current_step_data_descriptions
        return data_descriptions


class ChoiceConfigSpaceBuilder(CompositeConfigSpaceBuilder):

    def get_config_space(self):
        cs = self._element.get_hyperparameter_search_space()
        choice_parameter = cs.get_hyperparameter('__choice__')
        for name, node in self._children.items():
            sub_cs = node.get_config_space()
            cs.add_configuration_space(name, sub_cs, {'parent': choice_parameter, 'value': name})
        return cs


    def explore_data_flow(self, data_description):
        data_descriptions = []
        for name, node in self._children.items():
            data_description_copy = data_description.copy()
            data_description_copy.add_choice(node, name)
            current_step_data_descriptions = node.explore_data_flow(data_description_copy)
            data_descriptions.extend(current_step_data_descriptions)
        return data_descriptions


class PathTracker(object):

    def __init__(self):
        self._node_path = []
        self._choices = []

    def add_path(self, node):
        self._node_path.append(node)

    def add_choice(self, node, option):
        self._choices.append((node, option))

    def copy(self):
        data_description_path = PathTracker()
        data_description_path._node_path = copy(self._node_path)
        data_description_path._choices = copy(self._choices)
        return data_description_path


class ArtifactStorage(object):

    def __init__(self):
        self._node_by_artifact = {}

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

    def copy(self):
        artifact_storage = ArtifactStorage()
        artifact_storage._node_by_artifact = copy(self._node_by_artifact)
        return artifact_storage


class DataDescriptionBase(object):

    def __init__(self, path_tracker, artifact_storage):
        self._path_tracker = path_tracker
        self._artifact_storage = artifact_storage

    @property
    def is_valid(self):
        return False

    @property
    def is_incompatible(self):
        return False


class DataDescription(DataDescriptionBase):

    def __init__(self, path_tracker=None, artifact_storage=None):
        if not path_tracker:
            path_tracker = PathTracker()
        if not artifact_storage:
            artifact_storage = ArtifactStorage()
        super(DataDescription, self).__init__(path_tracker, artifact_storage)

    @property
    def is_valid(self):
        return True

    @property
    def is_incompatible(self):
        return False

    def copy(self):
        path_tracker = self._path_tracker.copy()
        artifact_storage = self._artifact_storage.copy()
        data_description = DataDescription(path_tracker, artifact_storage)
        return data_description

    def get_artifacts(self):
        return self._artifact_storage.get_artifacts()

    def update_artifacts(self, node, artifacts):
        return self._artifact_storage.update_artifacts(node, artifacts)

    def add_artifact(self, node, artifact):
        return self._artifact_storage.add_artifact(node, artifact)

    def remove_artifact(self, artifact):
        return self._artifact_storage.remove_artifact(artifact)

    def get_node_by_artifact(self, artifact):
        return self._artifact_storage.get_node_by_artifact(artifact)

    def add_path(self, node):
        return self._path_tracker.add_path(node)

    def add_choice(self, node, option):
        return self._path_tracker.add_choice(node, option)


class IncompatibleDataDescription(DataDescription):

    def __init__(self, data_description, node, artifacts):
        self.data_description = data_description
        self.node = node
        self.artifacts = artifacts
        path_tracker = data_description._path_tracker
        artifact_storage = data_description._artifact_storage
        super(IncompatibleDataDescription).__init__(path_tracker, artifact_storage)

    @property
    def is_valid(self):
        return False

    @property
    def is_incompatible(self):
        return True