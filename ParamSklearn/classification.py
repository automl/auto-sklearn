from collections import OrderedDict
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from ParamSklearn import components as components
from ParamSklearn.base import ParamSklearnBaseEstimator
from ParamSklearn.util import SPARSE
from ParamSklearn.components.preprocessing.balancing import Balancing
import ParamSklearn.create_searchspace_util


class ParamSklearnClassifier(ClassifierMixin, ParamSklearnBaseEstimator):
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
        :meth:`ParamSklearn.classification.ParamSklearnClassifier.fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`ParamSklearn.classification.ParamSklearnClassifier.fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        self.num_targets = 1 if len(y.shape) == 1 else y.shape[1]

        # Weighting samples has to be done here, not in the components
        if self.configuration['balancing:strategy'] == 'weighting':
            balancing = Balancing(strategy='weighting')
            init_params, fit_params = balancing.get_weights(
                y, self.configuration['classifier:__choice__'],
                self.configuration['preprocessor:__choice__'],
                init_params, fit_params)

        X, fit_params = super(ParamSklearnClassifier, self).pre_transform(
            X, y, fit_params=fit_params, init_params=init_params)

        return X, fit_params

    def predict_proba(self, X, batch_size=None):
        """predict_proba.

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
        """
        if batch_size is None:
            Xt = X
            for name, transform in self.pipeline_.steps[:-1]:
                Xt = transform.transform(Xt)

            return self.pipeline_.steps[-1][-1].predict_proba(Xt)

        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0].copy())

                # Binary or Multiclass
                if len(target) == 1:
                    y = np.zeros((X.shape[0], target.shape[1]))

                    for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                            batch_size)))):
                        batch_from = k * batch_size
                        batch_to = min([(k + 1) * batch_size, X.shape[0]])
                        y[batch_from:batch_to] = \
                            self.predict_proba(X[batch_from:batch_to],
                                               batch_size=None)

                elif len(target) > 1:
                    y = [np.zeros((X.shape[0], target[i].shape[1]))
                         for i in range(len(target))]

                    for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                            batch_size)))):
                        batch_from = k * batch_size
                        batch_to = min([(k + 1) * batch_size, X.shape[0]])
                        predictions = \
                            self.predict_proba(X[batch_from:batch_to],
                                               batch_size=None)

                        for i in range(len(target)):
                            y[i][batch_from:batch_to] = predictions[i]

                return y

    @classmethod
    def get_hyperparameter_search_space(cls, include=None, exclude=None,
                                        dataset_properties=None):
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        pipeline = cls._get_pipeline()
        cs = cls._get_hyperparameter_search_space(cs, dataset_properties,
                                                  exclude, include, pipeline)

        classifiers = cs.get_hyperparameter('classifier:__choice__').choices
        preprocessors = cs.get_hyperparameter('preprocessor:__choice__').choices
        available_classifiers = pipeline[-1][1].get_available_components(
            dataset_properties)
        available_preprocessors = pipeline[-2][1].get_available_components(
            dataset_properties)

        # A classifier which can handle sparse data after the densifier
        for key in classifiers:
            if SPARSE in available_classifiers[key].get_properties()['input']:
                if 'densifier' in preprocessors:
                    cs.add_forbidden_clause(
                        ForbiddenAndConjunction(
                            ForbiddenEqualsClause(
                                cs.get_hyperparameter(
                                    'classifier:__choice__'), key),
                            ForbiddenEqualsClause(
                                cs.get_hyperparameter(
                                    'preprocessor:__choice__'), 'densifier')
                        ))

        # which would take too long
        # Combinations of non-linear models with feature learning:
        classifiers_ = ["adaboost", "decision_tree", "extra_trees",
                        "gradient_boosting", "k_nearest_neighbors",
                        "libsvm_svc", "random_forest", "gaussian_nb",
                        "decision_tree"]
        feature_learning = ["kitchen_sinks", "nystroem_sampler"]

        for c, f in product(classifiers_, feature_learning):
            if c not in classifiers:
                continue
            if f not in preprocessors:
                continue
            try:
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        "classifier:__choice__"), c),
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        "preprocessor:__choice__"), f)))
            except KeyError:
                pass

        # Won't work
        # Multinomial NB etc does not work with negative values, don't use
        # it with standardization, features learning, pca
        classifiers_ = ["multinomial_nb", "bernoulli_nb"]
        preproc_with_negative_X = ["kitchen_sinks", "pca", "truncatedSVD",
                                   "fast_ica", "kernel_pca", "nystroem_sampler"]
        scaling_strategies = ['standard', 'none', "normalize"]
        for c in classifiers_:
            if c not in classifiers:
                continue
            for scaling_strategy in scaling_strategies:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "rescaling:strategy"), scaling_strategy),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "classifier:__choice__"), c)))
                except KeyError:
                    pass

        for c, f in product(classifiers_, preproc_with_negative_X):
            if c not in classifiers:
                continue
            if f not in preprocessors:
                continue
            try:
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        "preprocessor:__choice__"), f),
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        "classifier:__choice__"), c)))
            except KeyError:
                pass

        # Now try to add things for which we know that they don't work
        forbidden_hyperparameter_combinations = \
            [("preprocessor:select_percentile_classification:score_func", "chi2",
              "rescaling:strategy", "standard"),
             ("preprocessor:select_percentile_classification:score_func", "chi2",
              "rescaling:strategy", "normalize"),
             ("preprocessor:select_percentile_classification:score_func", "chi2",
              "rescaling:strategy", "none"),
             ("preprocessor:select_rates:score_func", "chi2",
              "rescaling:strategy", "standard"),
             ("preprocessor:select_rates:score_func", "chi2",
              "rescaling:strategy", "none"),
             ("preprocessor:select_rates:score_func", "chi2",
              "rescaling:strategy", "normalize"),
             ("preprocessor:nystroem_sampler:kernel", 'chi2', "rescaling:strategy",
              "standard"),
             ("preprocessor:nystroem_sampler:kernel", 'chi2', "rescaling:strategy",
              "normalize"),
             ("preprocessor:nystroem_sampler:kernel", 'chi2', "rescaling:strategy",
              "none")]
        for hp_name_1, hp_value_1, hp_name_2, hp_value_2 in \
                forbidden_hyperparameter_combinations:
            try:
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        hp_name_1), hp_value_1),
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        hp_name_2), hp_value_2)
                ))
            except:
                pass

        return cs

    @staticmethod
    def _get_pipeline():
        steps = []

        # Add the always active preprocessing components
        steps.extend(
            [["imputation",
              components.preprocessing._preprocessors['imputation']],
             ["rescaling",
              components.preprocessing._preprocessors['rescaling']],
             ["balancing",
              components.preprocessing._preprocessors['balancing']]])

        # Add the preprocessing component
        steps.append(['preprocessor',
                      components.preprocessing._preprocessors['preprocessor']])

        # Add the classification component
        steps.append(['classifier',
                      components.classification_components._classifiers['classifier']])
        return steps

    def _get_estimator_hyperparameter_name(self):
        return "classifier"

