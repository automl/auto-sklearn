from abc import ABCMeta, abstractmethod
from collections import defaultdict
import copy

import numpy as np
import sklearn
if sklearn.__version__ != "0.16.1":
    raise ValueError("ParamSklearn supports only sklearn version 0.16.1, "
                     "you installed %s." % sklearn.__version__)

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state, check_is_fitted

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition, AbstractConjunction

from . import components as components


class ParamSklearnBaseEstimator(BaseEstimator):
    """Base class for all ParamSklearn task models.

    Notes
    -----
    This class should not be instantiated, only subclassed."""
    __metaclass__ = ABCMeta

    def __init__(self, configuration, random_state=None):
        self.configuration = configuration

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

    def fit(self, X, y, fit_params=None, init_params=None):
        """Fit the selected algorithm to the training data.

        Parameters
        ----------
        X : array-like or sparse, shape = (n_samples, n_features)
            Training data. The preferred type of the matrix (dense or sparse)
            depends on the estimator selected.

        y : array-like
            Targets

        fit_params : dict
            See the documentation of sklearn.pipeline.Pipeline for formatting
            instructions.

        init_params : dict
            Pass arguments to the constructors of single methods. To pass
            arguments to only one of the methods (lets says the
            OneHotEncoder), seperate the class name from the argument by a ':'.

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
        X, fit_params = self.pre_transform(X, y, fit_params=fit_params,
                                          init_params=init_params)
        self.fit_estimator(X, y, fit_params=fit_params)
        return self

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        # Save all transformation object in a list to create a pipeline object
        steps = []

        # seperate the init parameters for the single methods
        init_params_per_method = defaultdict(dict)
        if init_params is not None and len(init_params) != 0:
            for init_param, value in init_params.items():
                method, param = init_param.split(":")
                init_params_per_method[method][param] = value

        # List of preprocessing steps (and their order)
        preprocessors_names = ["imputation", "rescaling",
                               self.configuration['preprocessor']]
        for preproc_name in preprocessors_names:
            preproc_params = {}

            for instantiated_hyperparameter in self.configuration:
                if not instantiated_hyperparameter.startswith(preproc_name):
                    continue
                if self.configuration[instantiated_hyperparameter] is None:
                    continue

                name_ = instantiated_hyperparameter.split(":")[1]
                preproc_params[name_] = self.configuration[
                    instantiated_hyperparameter]

            preproc_params.update(init_params_per_method[preproc_name])
            preprocessor_object = components.preprocessing_components. \
                _preprocessors[preproc_name](random_state=self.random_state,
                                             **preproc_params)
            steps.append((preproc_name, preprocessor_object))

        # Extract Estimator Hyperparameters from the configuration object
        estimator_name = self.configuration[
            self._get_estimator_hyperparameter_name()]
        estimator_parameters = {}
        for instantiated_hyperparameter in self.configuration:
            if not instantiated_hyperparameter.startswith(estimator_name):
                continue
            if self.configuration[instantiated_hyperparameter] is None:
                continue

            name_ = instantiated_hyperparameter.split(":")[1]
            estimator_parameters[name_] = self.configuration[
                instantiated_hyperparameter]

        estimator_parameters.update(init_params_per_method[estimator_name])
        estimator_object = self._get_estimator_components()[
            estimator_name](random_state=self.random_state,
                            **estimator_parameters)
        steps.append((estimator_name, estimator_object))

        self._validate_input_X(X)
        self._validate_input_Y(y)

        self.pipeline_ = Pipeline(steps)
        if fit_params is None or not isinstance(fit_params, dict):
            fit_params = dict()
        else:
            fit_params = {key.replace(":", "__"): value for key, value in
                          fit_params.items()}
        X, fit_params = self.pipeline_._pre_transform(X, y, **fit_params)
        return X, fit_params

    def fit_estimator(self, X, y, fit_params=None):
        check_is_fitted(self, 'pipeline_')
        if fit_params is None:
            fit_params = {}
        self.pipeline_.steps[-1][-1].fit(X, y, **fit_params)
        return self

    def iterative_fit(self, X, y, fit_params=None, n_iter=1):
        check_is_fitted(self, 'pipeline_')
        if fit_params is None:
            fit_params = {}
        self.pipeline_.steps[-1][-1].iterative_fit(X, y, n_iter=n_iter,
                                                   **fit_params)

    def estimator_supports_iterative_fit(self):
        check_is_fitted(self, 'pipeline_')
        return hasattr(self.pipeline_.steps[-1][-1], 'iterative_fit')

    def configuration_fully_fitted(self):
        check_is_fitted(self, 'pipeline_')
        return self.pipeline_.steps[-1][-1].configuration_fully_fitted()

    def _validate_input_X(self, X):
        # TODO: think of all possible states which can occur and how to
        # handle them
        """
        if not self.pipeline_[-1].handles_missing_values() or \
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
        """
        pass

    def _validate_input_Y(self, Y):
        """
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
        """
        pass

    def add_model_class(self, model):
        """
        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError()

    def predict(self, X, batch_size=None):
        """Predict the classes using the selected model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the ParamSklearn pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Returns the predicted values"""
        # TODO check if fit() was called before...

        if batch_size is None:
            self._validate_input_X(X)
            return self.pipeline_.predict(X)
        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                if self.num_targets == 1:
                    y = np.zeros((X.shape[0],))
                else:
                    y = np.zeros((X.shape[0], self.num_targets))

                # Copied and adapted from the scikit-learn GP code
                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                                                  batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to], batch_size=None)

                return y

    @classmethod
    def get_hyperparameter_search_space(cls, estimator_name,
                                         default_estimator,
                                         estimator_components,
                                         default_preprocessor,
                                         preprocessor_components,
                                         dataset_properties,
                                         always_active):
        """Return the configuration space for the CASH problem.

        This method should be called by the method
        get_hyperparameter_search_space of a subclass. After the subclass
        assembles a list of available estimators and preprocessor components,
        _get_hyperparameter_search_space can be called to do the work of
        creating the actual
        HPOlibConfigSpace.configuration_space.ConfigurationSpace object.

        Parameters
        ----------
        estimator_name : str
            Name of the estimator hyperparameter which will be used in the
            configuration space. For a classification task, this would be
            'classifier'.

        estimator_components : dict {name: component}
            Dictionary with all estimator components to be included in the
            configuration space.

        preprocessor_components : dict {name: component}
            Dictionary with all preprocessor components to be included in the
            configuration space. .

        always_active : list of str
            A list of components which will always be active in the pipeline.
            This is useful for components like imputation which have
            hyperparameters to be configured, but which do not have any parent.

        default_estimator : str
            Default value for the estimator hyperparameter.

        Returns
        -------
        cs : HPOlibConfigSpace.configuration_space.Configuration
            The configuration space describing the ParamSklearnClassifier.

        """

        cs = ConfigurationSpace()

        available_estimators = estimator_components
        available_preprocessors = preprocessor_components

        if default_estimator is None:
            default_estimator = available_estimators.keys()[0]

        estimator = CategoricalHyperparameter(estimator_name,
            available_estimators.keys(), default=default_estimator)
        cs.add_hyperparameter(estimator)
        for name in available_estimators.keys():

            # We have to retrieve the configuration space every time because
            # we change the objects it returns. If we reused it, we could not
            # retrieve the conditions further down
            # TODO implement copy for hyperparameters and forbidden and
            # conditions!

            estimator_configuration_space = available_estimators[name]. \
                get_hyperparameter_search_space(dataset_properties)
            for parameter in estimator_configuration_space.get_hyperparameters():
                new_parameter = copy.deepcopy(parameter)
                new_parameter.name = "%s:%s" % (name, new_parameter.name)
                cs.add_hyperparameter(new_parameter)
                # We must only add a condition if the hyperparameter is not
                # conditional on something else
                if len(estimator_configuration_space.
                        get_parents_of(parameter)) == 0:
                    condition = EqualsCondition(new_parameter, estimator, name)
                    cs.add_condition(condition)

            for condition in available_estimators[name]. \
                    get_hyperparameter_search_space(dataset_properties).get_conditions():
                dlcs = condition.get_descendant_literal_conditions()
                for dlc in dlcs:
                    if not dlc.child.name.startswith(name):
                        dlc.child.name = "%s:%s" % (name, dlc.child.name)
                    if not dlc.parent.name.startswith(name):
                        dlc.parent.name = "%s:%s" % (name, dlc.parent.name)
                cs.add_condition(condition)

            for forbidden_clause in available_estimators[name]. \
                    get_hyperparameter_search_space(dataset_properties).forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                                                             dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)

        preprocessor_choices = filter(lambda app: app not in always_active,
                                      available_preprocessors.keys())
        preprocessor = CategoricalHyperparameter("preprocessor",
            preprocessor_choices, default=default_preprocessor)
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors.keys():
            preprocessor_configuration_space = available_preprocessors[name]. \
                get_hyperparameter_search_space(dataset_properties)
            for parameter in preprocessor_configuration_space.get_hyperparameters():
                new_parameter = copy.deepcopy(parameter)
                new_parameter.name = "%s:%s" % (name, new_parameter.name)
                cs.add_hyperparameter(new_parameter)
                # We must only add a condition if the hyperparameter is not
                # conditional on something else
                if len(preprocessor_configuration_space.
                        get_parents_of(
                        parameter)) == 0 and name not in always_active:
                    condition = EqualsCondition(new_parameter, preprocessor,
                                                name)
                    cs.add_condition(condition)

            for condition in available_preprocessors[name]. \
                    get_hyperparameter_search_space(dataset_properties).get_conditions():
                if not isinstance(condition, AbstractConjunction):
                    dlcs = [condition]
                else:
                    dlcs = condition.get_descendent_literal_conditions()
                for dlc in dlcs:
                    if not dlc.child.name.startswith(name):
                        dlc.child.name = "%s:%s" % (name, dlc.child.name)
                    if not dlc.parent.name.startswith(name):
                        dlc.parent.name = "%s:%s" % (name, dlc.parent.name)
                cs.add_condition(condition)

            for forbidden_clause in available_preprocessors[name]. \
                    get_hyperparameter_search_space(dataset_properties).forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                                                             dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)

        return cs

    @staticmethod
    def _get_estimator_hyperparameter_name():
        pass

    @staticmethod
    def _get_estimator_components():
        pass