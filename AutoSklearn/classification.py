import copy
from itertools import product

from sklearn.base import ClassifierMixin

from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from . import components as components
from .base import AutoSklearnBaseEstimator


class AutoSklearnClassifier(ClassifierMixin, AutoSklearnBaseEstimator):
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
    _pipeline = ["imputation", "rescaling", "__preprocessor__",
                 "__estimator__"]

    def predict_proba(self, X):
        """predict_proba.

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

        return self._pipeline.steps[-1][-1].predict_proba(Xt)

    @staticmethod
    def get_hyperparameter_search_space(include_estimators=None,
                                        exclude_estimators=None,
                                        include_preprocessors=None,
                                        exclude_preprocessors=None,
                                        multiclass=False,
                                        multilabel=False,
                                        sparse=False):

        if include_estimators is not None and exclude_estimators is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_estimators cannot be used together.")

        if include_preprocessors is not None and exclude_preprocessors is not None:
            raise ValueError("The arguments include_preprocessors and "
                             "exclude_preprocessors cannot be used together.")

        # Compile a list of all estimator objects for this problem
        available_classifiers = AutoSklearnClassifier._get_estimator_components()

        classifiers = dict()
        for name in available_classifiers:
            if include_estimators is not None and \
                            name not in include_estimators:
                continue
            elif exclude_estimators is not None and \
                            name in exclude_estimators:
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
            classifiers[name] = available_classifiers[name]

        if len(classifiers) == 0:
            raise ValueError("No classifier to build a configuration space "
                             "for...")

        # Hardcode the defaults based on some educated guesses
        classifier_defaults = ['random_forest', 'liblinear', 'sgd',
                               'libsvm_svc']
        classifier_default = None
        for cd_ in classifier_defaults:
            if cd_ in classifiers:
                classifier_default = cd_
                break
        if classifier_default is None:
            classifier_default = classifiers[0]

        # Compile a list of preprocessor for this problem
        available_preprocessors = \
            components.preprocessing_components._preprocessors

        preprocessors = dict()
        for name in available_preprocessors:
            if name in ["imputation", "rescaling"]:
                preprocessors[name] = available_preprocessors[name]
                continue
            elif include_preprocessors is not None and \
                            name not in include_preprocessors:
                continue
            elif exclude_preprocessors is not None and \
                            name in exclude_preprocessors:
                continue

            if multiclass is True and available_preprocessors[name]. \
                    get_properties()['handles_multiclass'] is False:
                continue
            if multilabel is True and available_preprocessors[name]. \
                    get_properties()['handles_multilabel'] is False:
                continue
            if sparse is True and available_preprocessors[name]. \
                    get_properties()['handles_sparse'] is False:
                continue

            preprocessors[name] = available_preprocessors[name]

        # Get the configuration space
        configuration_space = AutoSklearnBaseEstimator\
            ._get_hyperparameter_search_space(
            AutoSklearnClassifier._get_estimator_hyperparameter_name(),
            classifiers, preprocessors,
            AutoSklearnClassifier._pipeline, classifier_default)

        # And now add forbidden parameter configurations which would take too
        # long

        # Combinations of tree-based models with feature learning:
        classifiers_ = ["extra_trees", "gradient_boosting",
                        "k_nearest_neighbors", "libsvm_svc", "random_forest"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering"]

        for c, f in product(classifiers_, feature_learning_):
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f)))
            except:
                pass

        return configuration_space

    @staticmethod
    def _get_estimator_hyperparameter_name():
        return "classifier"

    @staticmethod
    def _get_estimator_components():
        return components.classification_components._classifiers