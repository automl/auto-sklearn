import copy
from itertools import product

import numpy as np
from sklearn.base import RegressorMixin

from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor

from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components import regression as \
    regression_components
from autosklearn.pipeline.components import feature_preprocessing as \
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
    config : ConfigSpace.configuration_space.Configuration
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
    def __init__(self, config=None, steps=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None):
        self._output_dtype = np.float32
        super().__init__(
            config=config, steps=steps,
            dataset_properties=dataset_properties,
            include=include, exclude=exclude, random_state=random_state,
            init_params=init_params)

    def fit_estimator(self, X, y, **fit_params):
        self.y_max_ = np.nanmax(y)
        self.y_min_ = np.nanmin(y)
        return super(SimpleRegressionPipeline, self).fit_estimator(
            X, y, **fit_params)

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        self.y_max_ = np.nanmax(y)
        self.y_min_ = np.nanmin(y)
        return super(SimpleRegressionPipeline, self).iterative_fit(
            X, y, n_iter=n_iter, **fit_params)

    def predict(self, X, batch_size=None):
        y = super().predict(X, batch_size=batch_size)
        y[y > (2 * self.y_max_)] = 2 * self.y_max_
        if self.y_min_ < 0:
            y[y < (2 * self.y_min_)] = 2 * self.y_min_
        elif self.y_min_ > 0:
            y[y < (0.5 * self.y_min_)] = 0.5 * self.y_min_
        return y

    def _get_hyperparameter_search_space(self, include=None, exclude=None,
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
        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'regression'
        if dataset_properties['target_type'] != 'regression':
            dataset_properties['target_type'] = 'regression'

        if 'sparse' not in dataset_properties:
            # This dataset is probably dense
            dataset_properties['sparse'] = False

        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        regressors = cs.get_hyperparameter('regressor:__choice__').choices
        preprocessors = cs.get_hyperparameter('feature_preprocessor:__choice__').choices
        available_regressors = self._final_estimator.get_available_components(
            dataset_properties)

        possible_default_regressor = copy.copy(list(
            available_regressors.keys()))
        default = cs.get_hyperparameter('regressor:__choice__').default_value
        del possible_default_regressor[
            possible_default_regressor.index(default)]

        # A regressor which can handle sparse data after the densifier is
        # forbidden for memory issues
        for key in regressors:
            if SPARSE in available_regressors[key].get_properties(dataset_properties=None)['input']:
                if 'densifier' in preprocessors:
                    while True:
                        try:
                            forb_reg = ForbiddenEqualsClause(
                                cs.get_hyperparameter('regressor:__choice__'), key)
                            forb_fpp = ForbiddenEqualsClause(cs.get_hyperparameter(
                                'feature_preprocessor:__choice__'), 'densifier')
                            cs.add_forbidden_clause(
                                ForbiddenAndConjunction(forb_reg, forb_fpp))
                            # Success
                            break
                        except ValueError:
                            # Change the default and try again
                            try:
                                default = possible_default_regressor.pop()
                            except IndexError:
                                raise ValueError(
                                    "Cannot find a legal default configuration.")
                            cs.get_hyperparameter(
                                'regressor:__choice__').default_value = default

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
                            "feature_preprocessor:__choice__"), f)))
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
                        'regressor:__choice__').default_value = default

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _get_estimator_components(self):
        return regression_components._regressors

    def _get_pipeline_steps(self, dataset_properties, init_params=None):
        steps = []

        default_dataset_properties = {'target_type': 'regression'}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ['data_preprocessing',
                DataPreprocessor(dataset_properties=default_dataset_properties)],
            ['feature_preprocessor',
                feature_preprocessing_components.FeaturePreprocessorChoice(
                    default_dataset_properties)],
            ['regressor',
                regression_components.RegressorChoice(default_dataset_properties)]
        ])

        return steps

    def _get_estimator_hyperparameter_name(self):
        return "regressor"
