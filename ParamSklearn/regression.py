from collections import defaultdict
import copy
from itertools import product

import sklearn
if sklearn.__version__ != "0.15.2":
    raise ValueError("ParamSklearn supports only sklearn version 0.15.2, "
                     "you installed %s." % sklearn.__version__)

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    InactiveHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from . import components as components
from .base import ParamSklearnBaseEstimator


class ParamSklearnRegressor(RegressorMixin, ParamSklearnBaseEstimator):
    """This class implements the regression task.

    It implements a pipeline, which includes one preprocessing step and one
    regression algorithm. It can render a search space including all known
    regression and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available regressors at runtime. For this reason the user must
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
    _estimator : The underlying scikit-learn regression model. This
        variable is assigned after a call to the
        :meth:`ParamSklearn.regression.ParamSklearnRegressor.fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`ParamSklearn.regression.ParamSklearnRegressor.fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """
    _pipeline = ["imputation", "rescaling", "__preprocessor__",
                 "__estimator__"]

    def _validate_input_X(self, X):
        # TODO: think of all possible states which can occur and how to
        # handle them
        pass

    def _validate_input_Y(self, Y):
        pass

    def add_model_class(self, model):
        """
        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError()

    @classmethod
    def get_hyperparameter_search_space(cls, include_estimators=None,
                                        exclude_estimators=None,
                                        include_preprocessors=None,
                                        exclude_preprocessors=None,
                                        dataset_properties=None):
        """Return the configuration space for the CASH problem.

        Parameters
        ----------
        include_estimators : list of str
            If include_estimators is given, only the regressors specified
            are used. Specify them by their module name; e.g., to include
            only the SVM use :python:`include_regressors=['svr']`.
            Cannot be used together with :python:`exclude_regressors`.

        exclude_estimators : list of str
            If exclude_estimators is given, only the regressors specified
            are used. Specify them by their module name; e.g., to include
            all regressors except the SVM use
            :python:`exclude_regressors=['svr']`.
            Cannot be used together with :python:`include_regressors`.

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
            The configuration space describing the ParamSklearnClassifier.
        """
        if include_estimators is not None and exclude_estimators is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_regressors cannot be used together.")

        if include_preprocessors is not None and exclude_preprocessors is not None:
            raise ValueError("The arguments include_preprocessors and "
                             "exclude_preprocessors cannot be used together.")

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()

        # Compile a list of all estimator objects for this problem
        available_regressors = ParamSklearnRegressor._get_estimator_components()

        # We assume that there exists only a single regression task. which
        # is different to classification where we have multiclass,
        # multilabel, etc
        regressors = dict()
        for name in available_regressors:
            if include_estimators is not None and \
                            name not in include_estimators:
                continue
            elif exclude_estimators is not None and \
                            name in exclude_estimators:
                continue
            if dataset_properties.get('sparse') is True and \
                    available_regressors[name].get_properties()[
                        'handles_sparse'] is False:
                continue
            regressors[name] = available_regressors[name]

        if len(regressors) == 0:
            raise ValueError("No regressors to build a configuration space "
                             "for...")

        # Hardcode the defaults based on some educated guesses
        classifier_defaults = ['random_forest', 'liblinear', 'sgd',
                               'libsvm_svc']
        regressor_default = None
        for cd_ in classifier_defaults:
            if cd_ in regressors:
                regressor_default = cd_
                break
        if regressor_default is None:
            regressor_default = regressors.keys()[0]

        # Compile a list of preprocessor for this problem
        available_preprocessors = \
            components.preprocessing_components._preprocessors

        preprocessors = dict()
        for name in available_preprocessors:
            if name in ParamSklearnRegressor._pipeline:
                preprocessors[name] = available_preprocessors[name]
                continue
            elif include_preprocessors is not None and \
                            name not in include_preprocessors:
                continue
            elif exclude_preprocessors is not None and \
                            name in exclude_preprocessors:
                continue

            if dataset_properties.get('sparse') is True and \
                    available_preprocessors[name].get_properties()[
                                'handles_sparse'] is False:
                continue
            elif dataset_properties.get('sparse') is False and \
                    available_preprocessors[name].get_properties()[
                                'handles_dense'] is False:
                continue
            elif available_preprocessors[name]. \
                    get_properties()['handles_regression'] is False:
                continue

            preprocessors[name] = available_preprocessors[name]

        # Get the configuration space
        configuration_space = super(ParamSklearnRegressor, cls).\
            get_hyperparameter_search_space(
            cls._get_estimator_hyperparameter_name(),
            regressor_default, regressors, preprocessors, dataset_properties,
            cls._pipeline, )

        # And now add forbidden parameter configurations which would take too
        # long

        # Combinations of tree-based models with feature learning:
        regressors_ = ["random_forest", "gradient_boosting", "gaussian_process"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering"]

        for c, f in product(regressors_, feature_learning_):
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "regressor"), c),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f)))
            except:
                pass

        return configuration_space

    @staticmethod
    def _get_estimator_components():
        return components.regression_components._regressors

    @staticmethod
    def _get_estimator_hyperparameter_name():
        return "regressor"