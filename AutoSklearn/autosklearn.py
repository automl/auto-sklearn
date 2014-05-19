import numpy as np
from numpy import float64

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import safe_asarray, assert_all_finite

from .components import classification as classification_components
from .components import preprocessing as preprocessing_components
from .util import NoModelException, hp_choice

task_types = set(["classification"])

class AutoSklearnClassifier(BaseEstimator, ClassifierMixin):
    """AutoSklearn

    AutoSklearn provides a search space covering a (work in progress) huge
    part of the scikit-learn models and the possibility to evaluate them.
    Together with a hyperparameter optimization package, AutoSklearn solves
    the Combined algorithm selection and Hyperparameter optimization problem
    (CASH).

    This class implements the classification task. It can perform
    preprocessing. It can render a search space for all known classification
    and preprocessing problems.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
    specifiy the parameters via set_params.

    Parameters
    ----------
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
       used by `np.random`.

    Attributes
    ----------
    _estimator : An underlying scikit-learn target model specified by a call to
        set_parames

    See also
    --------

    References
    ----------

    Examples
    --------
    """
    def __init__(self,
                 classifier=None,
                 preprocessor=None,
                 random_state=None,
                 parameters=None):

        # Test that either the classifier or the parameters
        if classifier is not None:
            assert parameters is None
            # TODO: Somehow assemble a parameters dictionary

        if preprocessor is not None:
            assert classifier is not None
            assert parameters is None

        if parameters is not None:
            assert classifier is None
            assert preprocessor is None
            classifier = parameters.get("classifier")
            preprocessor = parameters.get("preprocessor")
            if preprocessor == "None":
                preprocessor = None

        self.random_state = random_state
        self._estimator = None
        self._preprocessor = None
        self.parameters = parameters if parameters is not None else {}
        # TODO: add valid parameters to the parameters dictionary

        # TODO: make sure that there are no duplicate classifiers
        self._available_classifiers = classification_components._classifiers
        self._available_preprocessors = preprocessing_components._preprocessors

        if random_state is None:
            random_state = check_random_state(1)

        self._estimator_class = self._available_classifiers.get(classifier)
        if classifier is not None and self._estimator_class is None:
            raise KeyError("The classifier %s is not in the list "
                           "of classifiers found on this system: %s" %
                           (classifier, self._available_classifiers))

        self._preprocessor_class = self._available_preprocessors.get(preprocessor)
        if preprocessor is not None and self._preprocessor_class is None:
            raise KeyError("The preprocessor %s is not in the list "
                           "of preprocessors found on this system: %s" %
                           (preprocessor, self._available_preprocessors))

    def fit(self, X, Y):
        # TODO: perform input validation
        # TODO: look if X.shape[0] == y.shape[0]
        # TODO: check if the hyperparameters have been set...
        if self._estimator_class is None:
            raise NoModelException(self, "fit(X, Y)")

        # Extract Hyperparameters from the parameters dict...
        space = self._estimator_class.get_hyperparameter_search_space()
        name = space["name"]

        parameters = {}
        for key in space:
            if "%s:%s" % (name, key) in self.parameters:
                parameters[key] = self.parameters["%s:%s" % (name, key)]

        random_state = check_random_state(self.random_state)
        self._estimator = self._estimator_class(random_state=random_state,
                                                **parameters)

        self._validate_input_X(X)
        self._validate_input_Y(Y)

        if self._preprocessor_class is not None:
            # TODO: copy everything or not?
            parameters = {}
            preproc_space = self._preprocessor_class\
                    .get_hyperparameter_search_space()
            preproc_name = preproc_space["name"]

            for key in preproc_space:
                if "%s:%s" % (preproc_name, key) in self.parameters:
                    parameters[key] = self.parameters["%s:%s" % (preproc_name, key)]

            self._preprocessor = self._preprocessor_class(
                random_state=random_state, **parameters)
            self._preprocessor.fit(X, Y)
            X = self._preprocessor.transform(X)

        self._estimator.fit(X, Y)
        return self

    def predict(self, X):
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
        raise NotImplementedError()

    def get_hyperparameter_search_space(self):
        classifiers = {}
        for name in self._available_classifiers:
            classifier_parameters = self._available_classifiers[name]\
                .get_hyperparameter_search_space()
            print classifier_parameters
            classifier_parameters["name"] = name
            classifiers["classifier:" + name] = classifier_parameters

        preprocessors = {}
        preprocessors[None] = {}
        for name in self._available_preprocessors:
            preprocessor_parameters = self._available_preprocessors[name]\
                .get_hyperparameter_search_space()
            preprocessor_parameters["name"] = name
            preprocessors["preprocessing:" + name] = preprocessor_parameters
        return {"classifier": hp_choice("classifier", classifiers.values()),
                "preprocessing": hp_choice("preprocessing", preprocessors.values())}

    # TODO: maybe provide an interface to the underlying predictor like
    # decision_function or predict_proba