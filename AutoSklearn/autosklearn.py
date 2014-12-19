from collections import defaultdict
import copy

import sklearn
if sklearn.__version__ != "0.15.2":
    raise ValueError("AutoSklearn supports only sklearn version 0.15.2, "
                     "you installed %s." % sklearn.__version__)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import safe_asarray, assert_all_finite

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    InactiveHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition

from . import components as components


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
        :meth:`AutoSklearn.autosklearn.AutoSklearnClassifier.fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`AutoSklearn.autosklearn.AutoSklearnClassifier.fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """
    def __init__(self, configuration, random_state=None):

        # TODO check sklearn version!
        self.configuration = configuration

        cs = self.get_hyperparameter_search_space()
        cs.check_configuration(configuration)

        self._pipeline = None

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

    def fit(self, X, Y, fit_params=None, init_params=None):
        """Fit the selected algorithm to the training data.

        Parameters
        ----------
        X : array-like or sparse, shape = (n_samples, n_features)
            Training data. The preferred type of the matrix (dense or sparse)
            depends on the classifier selected.

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
        # TODO: this is an example of the antipattern of not properly
        #       initializing a class in the init function!
        # TODO: can this happen now that a configuration is specified at
        # instantiation time

        steps = []
        init_params_per_method = defaultdict(dict)
        if init_params is not None:
            for init_param, value in init_params:
                method, param = init_param.split(":")
                init_params_per_method[method][param] = value

        preprocessors_names = ["imputation", "rescaling",
                               self.configuration['preprocessor'].value]

        for preproc_name in preprocessors_names:
            if preproc_name != "None":
                preproc_params = {}

                for instantiated_hyperparameter in self.configuration:
                    if not instantiated_hyperparameter.hyperparameter.name \
                            .startswith(preproc_name):
                        continue
                    if isinstance(instantiated_hyperparameter,
                                  InactiveHyperparameter):
                        continue

                    name_ = instantiated_hyperparameter.hyperparameter.name. \
                        split(":")[1]
                    preproc_params[name_] = instantiated_hyperparameter.value

                preproc_params.update(init_params_per_method[preproc_name])
                preprocessor_object = components.preprocessing_components. \
                    _preprocessors[preproc_name](random_state=self.random_state,
                                                 **preproc_params)
                steps.append((preproc_name, preprocessor_object))

        # Extract Hyperparameters from the configuration object
        classifier_name = self.configuration["classifier"].value
        classifier_parameters = {}
        for instantiated_hyperparameter in self.configuration:
            if not instantiated_hyperparameter.hyperparameter.name.startswith(
                    classifier_name):
                continue
            if isinstance(instantiated_hyperparameter, InactiveHyperparameter):
                continue

            name_ = instantiated_hyperparameter.hyperparameter.name.\
                split(":")[1]
            classifier_parameters[name_] = instantiated_hyperparameter.value

        classifier_parameters.update(init_params_per_method[classifier_name])
        classifier_object = components.classification_components._classifiers\
            [classifier_name](random_state=self.random_state,
                              **classifier_parameters)
        steps.append((classifier_name, classifier_object))

        self._validate_input_X(X)
        self._validate_input_Y(Y)

        self._pipeline = Pipeline(steps)
        self._pipeline.fit(X, Y)
        return self

    def predict(self, X):
        """Predict the classes using the selected model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Returns the predicted values"""
        # TODO check if fit() was called before...
        self._validate_input_X(X)
        return self._pipeline.predict(X)

    def scores(self, X):
        """Predict confidence scores for samples using the selected model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        self._validate_input_X(X)

        Xt = X
        for name, transform in self._pipeline.steps[:-1]:
            Xt = transform.transform(Xt)
        return self._pipeline.steps[-1][-1].scores(Xt)

    def _validate_input_X(self, X):
        # TODO: think of all possible states which can occur and how to
        # handle them
        """
        if not self._pipeline[-1].handles_missing_values() or \
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

    @staticmethod
    def get_hyperparameter_search_space(include_classifiers=None,
                                        exclude_classifiers=None,
                                        include_preprocessors=None,
                                        exclude_preprocessors=None,
                                        multiclass=False,
                                        multilabel=False,
                                        sparse=False):
        """Return the configuration space for the CASH problem.

        Parameters
        ----------
        include_classifiers : list of str
            If include_classifiers is given, only the classifiers specified
            are used. Specify them by their module name; e.g., to include
            only the SVM use :python:`include_classifiers=['libsvm_svc']`.
            Cannot be used together with :python:`exclude_classifiers`.

        exclude_classifiers : list of str
            If exclude_classifiers is given, only the classifiers specified
            are used. Specify them by their module name; e.g., to include
            all classifiers except the SVM use
            :python:`exclude_classifiers=['libsvm_svc']`.
            Cannot be used together with :python:`include_classifiers`.

        include_preprocessors : list of str
            If include_preprocessors is given, only the preprocessors specified
            are used. Specify them by their module name; e.g., to include
            only the PCA use :python:`include_preprocessors=['pca']`.
            Cannot be used together with :python:`exclude_preprocessors`.

        exclude_preprocessors : list of str
            If include_preprocessors is given, only the preprocessors specified
            are used. Specify them by their module name; e.g., to include
            all preprocessors except the PCA use
            :python:`exclude_preprocessors=['pca']`.
            Cannot be used together with :python:`include_preprocessors`.

        Returns
        -------
        cs : HPOlibConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.

        """
        if include_classifiers is not None and exclude_classifiers is not None:
            raise ValueError("The arguments include_classifiers and "
                             "exclude_classifiers cannot be used together.")

        if include_preprocessors is not None and exclude_preprocessors is not None:
            raise ValueError("The arguments include_preprocessors and "
                             "exclude_preprocessors cannot be used together.")

        always_active = ["imputation", "rescaling"]

        cs = ConfigurationSpace()

        available_classifiers = \
            components.classification_components._classifiers
        available_preprocessors = \
            components.preprocessing_components._preprocessors

        names = []
        names_ = []
        for name in available_classifiers:
            if name in always_active:
                names_.append(name)
                continue
            elif include_classifiers is not None and \
                            name not in include_classifiers:
                continue
            elif exclude_classifiers is not None and \
                            name in exclude_classifiers:
                continue

            if multiclass is True and available_classifiers[name]. \
                    get_properties()['handles_multiclass'] is False:
                continue
            if multilabel is True and available_classifiers[name]. \
                    get_properties()['handles_multilabel'] is False:
                continue
            if sparse is True and available_classifiers[name]. \
                    get_properties()['handles_sparse'] is False:
                continue
            names.append(name)

        if len(names + names_) == 0:
            raise ValueError("No classifier to build a configuration space "
                             "for...")

        classifier = CategoricalHyperparameter("classifier", names,
            default='random_forest' if 'random_forest' in names else names[0])
        cs.add_hyperparameter(classifier)
        for name in names + names_:

            # We have to retrieve the configuration space every time because
            # we change the objects it returns. If we reused it, we could not
            #  retrieve the conditions further down
            # TODO implement copy for hyperparameters and forbidden and
            # conditions!

            classifier_configuration_space = available_classifiers[name]. \
                get_hyperparameter_search_space()
            for parameter in classifier_configuration_space.get_hyperparameters():
                new_parameter = copy.deepcopy(parameter)
                new_parameter.name = "%s:%s" % (name, new_parameter.name)
                cs.add_hyperparameter(new_parameter)
                # We must only add a condition if the hyperparameter is not
                # conditional on something else
                if len(classifier_configuration_space.
                        get_parents_of(parameter)) == 0:
                    condition = EqualsCondition(new_parameter, classifier, name)
                    cs.add_condition(condition)

            for condition in available_classifiers[name]. \
                    get_hyperparameter_search_space().get_conditions():
                dlcs = condition.get_descendant_literal_conditions()
                for dlc in dlcs:
                    if not dlc.child.name.startswith(name):
                        dlc.child.name = "%s:%s" % (name, dlc.child.name)
                    if not dlc.parent.name.startswith(name):
                        dlc.parent.name = "%s:%s" % (name, dlc.parent.name)
                cs.add_condition(condition)

            for forbidden_clause in available_classifiers[name]. \
                    get_hyperparameter_search_space().forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.name.startswith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                            dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)


        names = []
        names_ = []
        for name in available_preprocessors:
            if name in always_active:
                names_.append(name)
                continue
            elif include_preprocessors is not None and \
                            name not in include_preprocessors:
                continue
            elif exclude_preprocessors is not None and \
                            name in exclude_preprocessors:
                continue
            names.append(name)

        preprocessor = CategoricalHyperparameter("preprocessor",
                                                 ["None"] + names,
                                                 default='None')
        cs.add_hyperparameter(preprocessor)
        for name in names + names_:
            preprocessor_configuration_space = available_preprocessors[name]. \
                get_hyperparameter_search_space()
            for parameter in preprocessor_configuration_space.get_hyperparameters():
                new_parameter = copy.deepcopy(parameter)
                new_parameter.name = "%s:%s" % (name, new_parameter.name)
                cs.add_hyperparameter(new_parameter)
                # We must only add a condition if the hyperparameter is not
                # conditional on something else
                if len(preprocessor_configuration_space.
                        get_parents_of(parameter)) == 0 and name not in always_active:
                    condition = EqualsCondition(new_parameter, preprocessor, name)
                    cs.add_condition(condition)

            for condition in available_preprocessors[name]. \
                    get_hyperparameter_search_space().get_conditions():
                dlcs = condition.get_descendent_literal_conditions()
                for dlc in dlcs:
                    if not dlc.child.name.startswith(name):
                        dlc.child.name = "%s:%s" % (name, dlc.child.name)
                    if not dlc.parent.name.startswith(name):
                        dlc.parent.name = "%s:%s" % (name, dlc.parent.name)
                cs.add_condition(condition)

            for forbidden_clause in available_preprocessors[name]. \
                    get_hyperparameter_search_space().forbidden_clauses:
                dlcs = forbidden_clause.get_descendant_literal_clauses()
                for dlc in dlcs:
                    if not dlc.hyperparameter.startwith(name):
                        dlc.hyperparameter.name = "%s:%s" % (name,
                            dlc.hyperparameter.name)
                cs.add_forbidden_clause(forbidden_clause)

        return cs

    # TODO: maybe provide an interface to the underlying predictor like
    # decision_function or predict_proba