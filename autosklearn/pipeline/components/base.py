from collections import OrderedDict
import importlib
import inspect
import pkgutil
import sys

from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import ConfigurationSpace
from sklearn.utils import check_random_state

from autosklearn.pipeline.graph_based_config_space import ChoiceConfigSpaceBuilder, LeafNodeConfigSpaceBuilder


def find_components(package, directory, base_class):
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules(
            [directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
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

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
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

    def set_hyperparameters(self, configuration, init_params=None):
        params = configuration.get_dictionary()

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
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



class AutoSklearnClassificationAlgorithm(AutoSklearnComponent):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    def __init__(self):
        self.estimator = None
        self.properties = None

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,) or shape = (n_samples, n_labels)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict_proba(self, X):
        """Predict probabilities.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        raise NotImplementedError()

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator


class AutoSklearnPreprocessingAlgorithm(AutoSklearnComponent):
    """Provide an abstract interface for preprocessing algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    def __init__(self):
        self.preprocessor = None

    def transform(self, X):
        """The transform function calls the transform function of the
        underlying scikit-learn model and returns the transformed array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the transformed training data

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def get_preprocessor(self):
        """Return the underlying preprocessor object.

        Returns
        -------
        preprocessor : the underlying preprocessor object
        """
        return self.preprocessor


class AutoSklearnRegressionAlgorithm(AutoSklearnComponent):
    """Provide an abstract interface for regression algorithms in
    auto-sklearn.

    Make a subclass of this and put it into the directory
    `autosklearn/pipeline/components/regression` to make it available."""

    def __init__(self):
        self.estimator = None
        self.properties = None

    def predict(self, X):
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator


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



class AutoSklearnChoice(CompositeAutoSklearnComponent):

    def __init__(self, random_state=None):
        """
        Parameters
        ----------
        dataset_properties : dict
            Describes the dataset to work on, this can change the
            configuration space constructed by auto-sklearn. Mandatory
            properties are:
            * target_type: classification or regression


            Optional properties are:
            * multiclass: whether the dataset is a multiclass classification
              dataset.
            * multilabel: whether the dataset is a multilabel classification
              dataset
        """
        #self.configuration = self.get_hyperparameter_search_space(
        #    dataset_properties).get_default_configuration()

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        #self.set_hyperparameters(self.configuration)
        self.choice = None
        components = self.get_components()
        components = [(name, cls()) for (name, cls) in components.items()]
        super(AutoSklearnChoice, self).__init__(components)

    def get_components(cls):
        raise NotImplementedError()

    def get_available_components(self, dataset_properties=None,
                                 include=None,
                                 exclude=None):
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

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            # TODO maybe check for sparse?

            components_dict[name] = available_comp[name]

        return components_dict

    def set_hyperparameters(self, configuration, init_params=None):
        new_params = {}

        params = configuration.get_dictionary()
        choice = params['__choice__']
        del params['__choice__']

        for param, value in params.items():
            param = param.replace(choice, '').replace(':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, '').replace(':', '')
                new_params[param] = value

        new_params['random_state'] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def get_hyperparameter_search_space(self, dataset_properties=None,
                                        default=None,
                                        include=None,
                                        exclude=None):
        raise NotImplementedError()

    def fit(self, X, y, **kwargs):
        if kwargs is None:
            kwargs = {}
        return self.choice.fit(X, y, **kwargs)

    def predict(self, X):
        return self.choice.predict(X)

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()

        cs.add_hyperparameter(CategoricalHyperparameter(name="__choice__", choices=["linear", "square", "exponential"], default="linear"))
        return cs

    def get_config_space_builder(self):
        builder = ChoiceConfigSpaceBuilder(self)
        for name, component in self.components.items():
            child = component.get_config_space_builder()
            builder.add_child(name, child)
        return builder
