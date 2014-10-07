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
    """This class implements the classification task. It can perform
    preprocessing. It can render a search space including all known
    classification and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
    specifiy the parameters via set_params.

    The user can specify the hyperparameters of the AutoSklearnClassifier
    either by giving the classifier and the preprocessor argument or the
    parameters argument.

    Parameters
    ----------
    classifier: dict
        A dictionary which contains at least the name of the classification
        algorithm. It can also contain {parameter : value} pairs.

    preprocessor: dict
        A dictionary which contains at least the name of the preprocessing
        algorithm. It can also contain {parameter : value} pairs.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`.

    parameters: dict
        A dictionary which contains at least {'classifier' : name}. It can
        also contain the classifiers hyperparameters in the form of {name +
        ':hyperparametername' : value}. To also use a preprocessing algorithm
        you must specify {'preprocessing': name}, then you can also add its
        hyperparameters in the form {name + ':hyperparametername' : value}.

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
            preprocessor = parameters.get("preprocessing")
            if preprocessor == "None":
                preprocessor = None

        self.random_state = random_state
        self._estimator = None
        self._preprocessor = None
        self.parameters = parameters if parameters is not None else {}
        # TODO: add valid parameters to the parameters dictionary

        # TODO: make sure that there are no duplicate classifiers
        self._available_classifiers = classification_components._classifiers
        classifier_parameters = set()
        for _classifier in self._available_classifiers:
            accepted_hyperparameter_names = self._available_classifiers[_classifier] \
                .get_all_accepted_hyperparameter_names()
            name = self._available_classifiers[_classifier].get_hyperparameter_search_space()['name']
            for key in accepted_hyperparameter_names:
                classifier_parameters.add("%s:%s" % (name, key))

        self._available_preprocessors = preprocessing_components._preprocessors
        preprocessor_parameters = set()
        for _preprocessor in self._available_preprocessors:
            accepted_hyperparameter_names = self._available_preprocessors[_preprocessor] \
                .get_all_accepted_hyperparameter_names()
            name = self._available_preprocessors[_preprocessor].get_hyperparameter_search_space()['name']
            for key in accepted_hyperparameter_names:
                preprocessor_parameters.add("%s:%s" % (name, key))

        for parameter in self.parameters:
            if parameter not in classifier_parameters and \
                    parameter not in preprocessor_parameters and \
                    parameter not in ("preprocessing", "classifier", "name"):
                print "Classifier parameters %s" % str(classifier_parameters)
                print "Preprocessing parameters %s" % str(preprocessor_parameters)
                raise ValueError("Parameter %s is unknown." % parameter)

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        if classifier is not None and 'name' in classifier:
            self._estimator_class = self._available_classifiers.get(classifier['name'])
            if self._estimator_class is None:
                raise KeyError("The classifier %s is not in the list "
                               "of classifiers found on this system: %s" %
                               (classifier, self._available_classifiers))
        else:
            self._estimator_class = None

        if preprocessor is not None and 'name' in preprocessor:
            self._preprocessor_class = self._available_preprocessors.get(preprocessor['name'])
            if self._preprocessor_class is None:
                raise KeyError("The preprocessor %s is not in the list "
                               "of preprocessors found on this system: %s" %
                               (preprocessor, self._available_preprocessors))
        else:
            self._preprocessor_class = None

    def fit(self, X, Y):
        """Fit the selected algorithm to the training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

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
        if self._estimator_class is None:
            raise NoModelException(self, "fit(X, Y)")

        # Extract Hyperparameters from the parameters dict...
        #space = self._estimator_class.get_hyperparameter_search_space()
        space = self._estimator_class.get_all_accepted_hyperparameter_names()
        name = self._estimator_class.get_hyperparameter_search_space()['name']

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

    def get_hyperparameter_search_space(self):
        """Return the configuration space for the CASH problem.

        Returns
        -------
        cs : dict
            A dictionary with all hyperparameters as hyperopt.pyll objects.

        """
        classifiers = {}
        for name in self._available_classifiers:
            classifier_parameters = self._available_classifiers[name]\
                .get_hyperparameter_search_space()
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