from copy import copy
from ConfigSpace import ForbiddenAndConjunction
from ConfigSpace import ForbiddenEqualsClause

from autosklearn.pipeline.constants import SPARSE, DENSE


class ConfigSpaceBuilder(object):

    def __init__(self, element):
        self._element = element
        self._parent = None
        self._name = None

    def _set_name(self, name):
        self._name = name

    def _set_parent(self, parent):
        self._parent = parent

    def get_path(self):
        name = self._name
        parent_path = self._parent.get_path() if self._parent else None

        if parent_path and name:
            return parent_path + ':' + name
        elif parent_path:
            return parent_path
        elif name:
            return name
        else:
            return ""

    def get_config_space(self):
        pass

    def explore_data_flow(self, data_description):
        pass

    def get_incompatible_nodes(self, cs, dataset_properties):
        data_description = DataDescription()

        if 'is_sparse' in dataset_properties:
            if dataset_properties['is_sparse']:
                data_description.add_artifact(None, SPARSE)
            else:
                data_description.add_artifact(None, DENSE)

        data_descriptions = self.explore_data_flow(data_description)
        incompatible_nodes = [x for x in data_descriptions if x.is_incompatible]
        incompatible_node_choices = [x.get_choices() for x in incompatible_nodes]

        for incompatible_node_choice in incompatible_node_choices:
            hp_and_options = [(cs.get_hyperparameter(path), option) for path, option in incompatible_node_choice]
            clauses = [ForbiddenEqualsClause(hp, option) for hp, option in hp_and_options]
            conjunction = ForbiddenAndConjunction(*clauses)
            cs.add_forbidden_clause(conjunction)

        return incompatible_nodes

    def build(self, dataset_properties):
        cs = self.get_config_space()
        self.get_incompatible_nodes(cs, dataset_properties)

        return cs


class InvalidDataArtifactsException(Exception):

    def __init__(self, artifacts):
        self.artifacts = artifacts
        message = "Unable to transform data with: %s" % artifacts
        super(InvalidDataArtifactsException, self).__init__(message)


class PathTracker(object):

    def __init__(self):
        self._node_path = []
        self._choices = []

    def add_path(self, node):
        self._node_path.append(node)

    def add_choice(self, node, option):
        self._choices.append((node, option))

    def get_choices(self):
        return self._choices

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


class IncompatibleDataDescription(DataDescriptionBase):

    def __init__(self, data_description, node, artifacts):
        self._exception_node = node
        self._incompatible_artifacts = artifacts
        path_tracker = data_description._path_tracker
        artifact_storage = data_description._artifact_storage
        super(IncompatibleDataDescription, self).__init__(path_tracker, artifact_storage)

    @property
    def is_valid(self):
        return False

    @property
    def is_incompatible(self):
        return True

    def get_choices(self):
        choices = self._path_tracker.get_choices()
        choice_paths = [(node.get_path() + ":__choice__", choice) for node, choice in choices]
        return choice_paths
