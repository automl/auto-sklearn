import numpy as np
from numpy import float64

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import safe_asarray, assert_all_finite

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    InactiveHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition

from . import components as components
from .util import NoModelException

task_types = set(["classification"])

class AutoSklearnClassifier(BaseEstimator, ClassifierMixin):
    """This class implements the classification task.

    It implements a pipeline, which includes one preprocessing step and one
    classification algorithm. It can render a search space including all known
    classification and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    HPOlibConfigSpace.configuration_space.Configuration.

    Parameters
    ----------
    configuration : HPOlibConfigSpace.configuration_space.Configuration
        The configuration to evaluate.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    _estimator : The underlying scikit-learn classification model. This
        variable is assigned after a call to the
        :ref:`AutoSklearn.autosklearn.AutoSklearnClassifier.fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :ref:`AutoSklearn.autosklearn.AutoSklearnClassifier.fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """
    def __init__(self, configuration, random_state=None):

        # TODO check sklearn version!

        self.random_state = random_state
        self.configuration = configuration

        cs = self.get_hyperparameter_search_space()
        cs.check_configuration(configuration)

        self._estimator = None
        self._preprocessor = None

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

    def fit(self, X, Y):
        """Fit the selected algorithm to the training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data. All values must be in the range [0,1].

        y : array-like, shape = [n_samples]
            Targets

        Returns
        -------
        self : returns an instance of self.

        Raises
        ------
        NoModelException
            NoModelException is raised if fit() is called without specifying
            a classification algorithm first.
        """
        # TODO: perform input validation
        # TODO: look if X.shape[0] == y.shape[0]
        # TODO: check if the hyperparameters have been set...
        # TODO: this is an example of the antipattern of not properly
        #       initializing a class in the init function!
        # TODO: can this happen now that a configuration is specified at
        # instantiation time
        if "classifier" not in self.configuration:
            raise NoModelException(self, "fit(X, Y)")

        # Extract Hyperparameters from the configuration object
        name = self.configuration["classifier"].value

        parameters = {}
        for instantiated_hyperparameter in self.configuration:
            if not instantiated_hyperparameter.hyperparameter.name.startswith(
                    name):
                continue
            if isinstance(instantiated_hyperparameter, InactiveHyperparameter):
                continue

            print instantiated_hyperparameter.hyperparameter.name
            name_ = instantiated_hyperparameter.hyperparameter.name.\
                split(":")[1]
            parameters[name_] = instantiated_hyperparameter.value

        print parameters
        random_state = check_random_state(self.random_state)
        self._estimator = components.classification_components._classifiers\
            [name](random_state=random_state, **parameters)

        self._validate_input_X(X)
        self._validate_input_Y(Y)

        preprocessor = self.configuration['preprocessor']
        if preprocessor.value != "__None__":
            preproc_name = preprocessor.value
            preproc_params = {}

            for instantiated_hyperparameter in self.configuration:
                if not instantiated_hyperparameter.hyperparameter.name\
                        .startswith(preproc_name):
                    continue
                if isinstance(instantiated_hyperparameter, InactiveHyperparameter):
                    continue

                name_ = instantiated_hyperparameter.hyperparameter.name. \
                    split(":")[1]
                preproc_params[name_] = instantiated_hyperparameter.value

            print preproc_params
            self._preprocessor = components.preprocessing_components.\
                _preprocessors[preproc_name](random_state=random_state, **preproc_params)
            self._preprocessor.fit(X, Y)
            X = self._preprocessor.transform(X)

        self._estimator.fit(X, Y)
        return self

    def predict(self, X):
        """Predict the classes using the selected model..

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns the predicted values"""
        # TODO check if fit() was called before...
        if self._preprocessor is not None:
            X = self._preprocessor.transform(X)
        self._validate_input_X(X)
        return self._estimator.predict(X)

    def _validate_input_X(self, X):
        # TODO: think of all possible states which can occur and how to
        # handle them
        if not self._estimator.handles_missing_values() or \
                (self._preprocessor is not None and not\
                self._preprocessor.handles_missing_value()):
            assert_all_finite(X)
            X = safe_asarray(X)
        else:
            raise NotImplementedError()

        if not self._estimator.handles_nominal_features() or \
                (self._preprocessor is not None and not \
                 self._preprocessor.handles_nominal_features()):
            if X.dtype not in (np.float64, float64, np.float32, float):
                raise ValueError("Data type of X matrix is not float but %s!"
                                 % X.dtype)
        else:
            raise NotImplementedError()

        if not self._estimator.handles_numeric_features() or \
                (self._preprocessor is not None and not \
                 self._preprocessor.handles_numeric_features()):
            raise NotImplementedError()
        else:
            if X.dtype not in (np.float64, float64, np.float32, float):
                raise ValueError("Data type of X matrix is not float but %s!"
                                 % X.dtype)

    def _validate_input_Y(self, Y):
        Y = np.atleast_1d(Y)
        if not self._estimator.handles_non_binary_classes() or \
                (self._preprocessor is not None and not \
                 self._preprocessor.handles_non_binary_classes()):
            unique = np.unique(Y)
            if unique > 2:
                raise ValueError("Estimator %s which only handles binary "
                                 "classes cannot handle %d unique values" %
                                 (self._estimator, unique))
        else:
            pass

        if len(Y.shape) > 1:
            raise NotImplementedError()

    def add_model_class(self, model):
        """
        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space():
        """Return the configuration space for the CASH problem.

        Returns
        -------
        cs : HPOlibConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.

        """
        cs = ConfigurationSpace()

        available_classifiers = \
            components.classification_components._classifiers
        available_preprocessors = \
            components.preprocessing_components._preprocessors

        classifier = CategoricalHyperparameter(
            "classifier", [name for name in available_classifiers])
        cs.add_hyperparameter(classifier)

        for name in available_classifiers:
            # We have to retrieve the configuration space every time because
            # we change the objects it returns. If we reused it, we could not
            #  retrieve the conditions further down
            # TODO implement copy for hyperparameters and forbidden and
            # conditions!
            for parameter in available_classifiers[name].\
                    get_hyperparameter_search_space().get_hyperparameters():
                parameter.name = "%s:%s" % (name, parameter.name)
                cs.add_hyperparameter(parameter)
                condition = EqualsCondition(parameter, classifier, name)
                cs.add_condition(condition)

            for condition in available_classifiers[name]. \
                    get_hyperparameter_search_space().get_conditions():
                dlcs = condition.get_descendent_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                            dlc.hyperparameter.name)
                cs.add_condition(condition)

            for forbidden_clause in available_classifiers[name]. \
                    get_hyperparameter_search_space().forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                            dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)

        preprocessor = CategoricalHyperparameter(
            "preprocessor", [name for name in available_preprocessors] + [
                "__None__"])
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            for parameter in available_preprocessors[name].\
                    get_hyperparameter_search_space().get_hyperparameters():
                parameter.name = "%s:%s" % (name, parameter.name)
                cs.add_hyperparameter(parameter)
                condition = EqualsCondition(parameter, preprocessor, name)
                cs.add_condition(condition)

            for condition in available_preprocessors[name]. \
                    get_hyperparameter_search_space().get_conditions():
                dlcs = condition.get_descendent_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.startwith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                            dlc.hyperparameter.name)
                cs.add_condition(condition)

            for forbidden_clause in available_preprocessors[name]. \
                    get_hyperparameter_search_space().forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.startwith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                            dlc.hyperparameter.name)
                print forbidden_clause
                cs.add_forbidden_clause(forbidden_clause)

        return cs

    # TODO: maybe provide an interface to the underlying predictor like
    # decision_function or predict_proba