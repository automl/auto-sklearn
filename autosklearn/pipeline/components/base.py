import importlib
import inspect
import pkgutil
import sys
from collections import OrderedDict

from ConfigSpace import Configuration

from autosklearn.pipeline.components.config_space import InvalidDataArtifactsException, ConfigSpaceBuilder, \
    IncompatibleDataDescription
from autosklearn.pipeline.constants import *


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules(
            [directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(
                        obj) and base_class in obj.__bases__:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class ThirdPartyComponents(object):
    def __init__(self, base_class):
        self.base_class = base_class
        self.components = OrderedDict()

    def add_component(self, obj):
        if inspect.isclass(obj) and self.base_class in obj.__bases__:
            name = obj.__name__
            classifier = obj
        else:
            raise TypeError('add_component works only with a subclass of %s' %
                            str(self.base_class))

        properties = set(classifier.get_properties())
        should_be_there = {'shortname', 'name', 'handles_regression',
                           'handles_classification', 'handles_multiclass',
                           'handles_multilabel', 'is_deterministic',
                           'input', 'output'}
        for property in properties:
            if property not in should_be_there:
                raise ValueError('Property %s must not be specified for '
                                 'algorithm %s. Only the following properties '
                                 'can be specified: %s' %
                                 (property, name, str(should_be_there)))
        for property in should_be_there:
            if property not in properties:
                raise ValueError('Property %s not specified for algorithm %s')

        self.components[name] = classifier
        print(name, classifier)


class AutoSklearnComponent(object):

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of the underlying algorithm.

        Find more information at :ref:`get_properties`

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    def transform_data_description(self, artifacts):
        # default properties-based transformation

        properties = self.get_properties()

        if REGRESSION in artifacts and not properties['handles_regression']:
            raise InvalidDataArtifactsException(REGRESSION)
        if BINARY_CLASSIFICATION in artifacts and not properties['handles_classification']:
            raise InvalidDataArtifactsException(BINARY_CLASSIFICATION)
        if MULTICLASS_CLASSIFICATION in artifacts and not properties['handles_multiclass']:
            raise InvalidDataArtifactsException(MULTICLASS_CLASSIFICATION)
        if MULTILABEL_CLASSIFICATION in artifacts and not properties['handles_multilabel']:
            raise InvalidDataArtifactsException(MULTILABEL_CLASSIFICATION)

        input = properties['input']
        if DENSE in artifacts and not DENSE in input:
            raise InvalidDataArtifactsException(DENSE)
        elif SPARSE in artifacts and not SPARSE in input:
            raise InvalidDataArtifactsException(SPARSE)

        if UNSIGNED_DATA in artifacts and (not UNSIGNED_DATA in input or not SIGNED_DATA in input):
            raise InvalidDataArtifactsException(UNSIGNED_DATA)
        elif SIGNED_DATA in artifacts and not SIGNED_DATA in input:
            raise InvalidDataArtifactsException(SIGNED_DATA)

        output = properties['output']
        artifacts = set(artifacts)
        if PREDICTIONS in output:
            artifacts.add(PREDICTIONS)
        if not INPUT in output:
            artifacts.discard(SIGNED_DATA)
            artifacts.discard(UNSIGNED_DATA)
            artifacts.discard(DENSE)
            artifacts.discard(SPARSE)
        if UNSIGNED_DATA in output:
            artifacts.add(UNSIGNED_DATA)
            artifacts.discard(SIGNED_DATA)
        if SIGNED_DATA in output:
            artifacts.add(SIGNED_DATA)
            artifacts.discard(UNSIGNED_DATA)
        if DENSE in output:
            artifacts.add(DENSE)
            artifacts.discard(SPARSE)
        if SPARSE in output:
            artifacts.add(SPARSE)
            artifacts.discard(DENSE)

        return list(artifacts)

    def get_hyperparameter_search_space(self):
        """Return the configuration space of this classification algorithm.

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def fit(self, X, y):
        """The fit function calls the fit function of the underlying
        scikit-learn model and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,) or shape = (n_sample, n_labels)

        Returns
        -------
        self : returns an instance of self.
            Targets

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def set_hyperparameters(self, configuration):
        if isinstance(configuration, Configuration):
            configuration = configuration.get_dictionary()

        for param, value in configuration.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        return self

    def __str__(self):
        name = self.get_properties()['name']
        return "autosklearn.pipeline %s" % name

    def get_config_space(self):
        builder = self.get_config_space_builder()
        cs = builder.build()
        return cs

    def get_config_space_builder(self):
        cs = LeafNodeConfigSpaceBuilder(self)
        return cs


class LeafNodeConfigSpaceBuilder(ConfigSpaceBuilder):

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
