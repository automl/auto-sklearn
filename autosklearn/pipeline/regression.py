from collections import OrderedDict
import copy
from itertools import product

import numpy as np
from sklearn.base import RegressorMixin

from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.pipeline.components import regression as regression_components
from autosklearn.pipeline.components import data_preprocessing as \
    data_preprocessing_components
from autosklearn.pipeline.components import  feature_preprocessing as \
    feature_preprocessing_components
from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.constants import SPARSE


class SimpleRegressionPipeline(RegressorMixin, BasePipeline):
    """This class implements the regression task.

    It implements a pipeline, which includes one preprocessing step and one
    regression algorithm. It can render a search space including all known
    regression and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available regressors at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    ConfigSpace.configuration_space.Configuration.

    Parameters
    ----------
    configuration : ConfigSpace.configuration_space.Configuration
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
        :meth:`autosklearn.pipeline.regression.SimpleRegressionPipeline.fit`
        method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`autosklearn.pipeline.regression.SimpleRegressionPipeline.fit`
        method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """
    def __init__(self, configuration, random_state=None):
        self._output_dtype = np.float32
        super(SimpleRegressionPipeline, self).__init__(configuration,
                                                       random_state)

    def pre_transform(self, X, Y, fit_params=None, init_params=None):
        X, fit_params = super(SimpleRegressionPipeline, self).pre_transform(
            X, Y, fit_params=fit_params, init_params=init_params)
        self.num_targets = 1 if len(Y.shape) == 1 else Y.shape[1]
        return X, fit_params

    def fit_estimator(self, X, y, fit_params=None):
        self.y_max_ = np.nanmax(y)
        self.y_min_ = np.nanmin(y)
        if fit_params is None:
            fit_params = {}
        return super(SimpleRegressionPipeline, self).fit_estimator(
            X, y, **fit_params)

    def iterative_fit(self, X, y, fit_params=None, n_iter=1):
        self.y_max_ = np.nanmax(y)
        self.y_min_ = np.nanmin(y)
        if fit_params is None:
            fit_params = {}
        return super(SimpleRegressionPipeline, self).iterative_fit(
            X, y, n_iter=n_iter, **fit_params)

    def predict(self, X, batch_size=None):
        y = super(SimpleRegressionPipeline, self).\
            predict(X, batch_size=batch_size)
        y[y > (2 * self.y_max_)] = 2 * self.y_max_
        if self.y_min_ < 0:
            y[y < (2 * self.y_min_)] = 2 * self.y_min_
        elif self.y_min_ > 0:
            y[y < (0.5 * self.y_min_)] = 0.5 * self.y_min_
        return y

    @classmethod
    def get_available_components(cls, available_comp, data_prop, inc, exc):
        components_dict = OrderedDict()
        for name in available_comp:
            if inc is not None and name not in inc:
                continue
            elif exc is not None and name in exc:
                continue
            entry = available_comp[name]

            if not entry.get_properties()['handles_regression']:
                continue
            components_dict[name] = entry
        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, include=None, exclude=None,
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
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if not 'target_type' in dataset_properties:
            dataset_properties['target_type'] = 'regression'
        if dataset_properties['target_type'] != 'regression':
            dataset_properties['target_type'] = 'regression'

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        pipeline = cls._get_pipeline()
        cs = cls._get_hyperparameter_search_space(cs, dataset_properties,
                                                  exclude, include, pipeline)

        regressors = cs.get_hyperparameter('regressor:__choice__').choices
        preprocessors = cs.get_hyperparameter('preprocessor:__choice__').choices
        available_regressors = pipeline[-1][1].get_available_components(
            dataset_properties)
        available_preprocessors = pipeline[-2][1].get_available_components(
            dataset_properties)

        possible_default_regressor = copy.copy(list(
            available_regressors.keys()))
        default = cs.get_hyperparameter('regressor:__choice__').default
        del possible_default_regressor[
            possible_default_regressor.index(default)]

        # A regressor which can handle sparse data after the densifier
        for key in regressors:
            if SPARSE in available_regressors[key].get_properties(dataset_properties=None)['input']:
                if 'densifier' in preprocessors:
                    while True:
                        try:
                            cs.add_forbidden_clause(
                                ForbiddenAndConjunction(
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter(
                                            'regressor:__choice__'), key),
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter(
                                            'preprocessor:__choice__'), 'densifier')
                                ))
                            break
                        except ValueError:
                            # Change the default and try again
                            try:
                                default = possible_default_regressor.pop()
                            except IndexError:
                                raise ValueError(
                                    "Cannot find a legal default configuration.")
                            cs.get_hyperparameter(
                                'regressor:__choice__').default = default

        # which would take too long
        # Combinations of tree-based models with feature learning:
        regressors_ = ["adaboost", "decision_tree", "extra_trees",
                       "gaussian_process", "gradient_boosting",
                       "k_nearest_neighbors", "random_forest", "xgradient_boosting"]
        feature_learning_ = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]

        for r, f in product(regressors_, feature_learning_):
            if r not in regressors:
                continue
            if f not in preprocessors:
                continue
            while True:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "regressor:__choice__"), r),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "preprocessor:__choice__"), f)))
                    break
                except KeyError:
                    break
                except ValueError:
                    # Change the default and try again
                    try:
                        default = possible_default_regressor.pop()
                    except IndexError:
                        raise ValueError(
                            "Cannot find a legal default configuration.")
                    cs.get_hyperparameter(
                        'regressor:__choice__').default = default

        return cs

    @staticmethod
    def _get_estimator_components():
        return regression_components._regressors

    @classmethod
    def _get_pipeline(cls):
        steps = []

        # Add the always active preprocessing components
        steps.extend(
            [["one_hot_encoding",
              data_preprocessing_components._preprocessors['one_hot_encoding']],
            ["imputation",
             data_preprocessing_components._preprocessors['imputation']],
             ["rescaling",
              data_preprocessing_components._preprocessors['rescaling']]])

        # Add the preprocessing component
        steps.append(['preprocessor',
                      feature_preprocessing_components.FeaturePreprocessorChoice])

        # Add the classification component
        steps.append(['regressor',
                      regression_components.RegressorChoice])
        return steps

    def _get_estimator_hyperparameter_name(self):
        return "regressor"
