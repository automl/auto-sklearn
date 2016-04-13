import copy
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from autosklearn.pipeline.components import classification as \
    classification_components
from autosklearn.pipeline.components import data_preprocessing as \
    data_preprocessing_components
from autosklearn.pipeline.components import feature_preprocessing as \
    feature_preprocessing_components
from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.constants import SPARSE
from autosklearn.pipeline.components.data_preprocessing.balancing import Balancing


class SimpleClassificationPipeline(ClassifierMixin, BasePipeline):
    """This class implements the classification task.

    It implements a pipeline, which includes one preprocessing step and one
    classification algorithm. It can render a search space including all known
    classification and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
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
    _estimator : The underlying scikit-learn classification model. This
        variable is assigned after a call to the
        :meth:`autosklearn.pipeline.classification.SimpleClassificationPipeline
        .fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`autosklearn.pipeline.classification.SimpleClassificationPipeline
        .fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def __init__(self, configuration, random_state=None):
        self._output_dtype = np.int32
        super(SimpleClassificationPipeline, self).__init__(configuration,
                                                           random_state)

    def pre_transform(self, X, y, fit_params=None, init_params=None):
        self.num_targets = 1 if len(y.shape) == 1 else y.shape[1]

        # Weighting samples has to be done here, not in the components
        if self.configuration['balancing:strategy'] == 'weighting':
            balancing = Balancing(strategy='weighting')
            init_params, fit_params = balancing.get_weights(
                y, self.configuration['classifier:__choice__'],
                self.configuration['preprocessor:__choice__'],
                init_params, fit_params)

        X, fit_params = super(SimpleClassificationPipeline, self).pre_transform(
            X, y, fit_params=fit_params, init_params=init_params)

        return X, fit_params

    def predict_proba(self, X, batch_size=None):
        """predict_proba.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the pipeline will be
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
                target = self.predict_proba(X[0:2].copy())

                y = np.zeros((X.shape[0], target.shape[1]),
                             dtype=np.float32)

                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                        batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict_proba(X[batch_from:batch_to],
                                           batch_size=None).\
                            astype(np.float32)

                return y

    @classmethod
    def get_hyperparameter_search_space(cls, include=None, exclude=None,
                                        dataset_properties=None):
        """Create the hyperparameter configuration space.

        Parameters
        ----------
        include : dict (optional, default=None)

        Returns
        -------
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if not 'target_type' in dataset_properties:
            dataset_properties['target_type'] = 'classification'
        if dataset_properties['target_type'] != 'classification':
            dataset_properties['target_type'] = 'classification'

        pipeline = cls._get_pipeline()
        cs = cls._get_hyperparameter_search_space(cs, dataset_properties,
                                                  exclude, include, pipeline)

        classifiers = cs.get_hyperparameter('classifier:__choice__').choices
        preprocessors = cs.get_hyperparameter('preprocessor:__choice__').choices
        available_classifiers = pipeline[-1][1].get_available_components(
            dataset_properties)
        available_preprocessors = pipeline[-2][1].get_available_components(
            dataset_properties)

        possible_default_classifier = copy.copy(list(
            available_classifiers.keys()))
        default = cs.get_hyperparameter('classifier:__choice__').default
        del possible_default_classifier[possible_default_classifier.index(default)]

        # A classifier which can handle sparse data after the densifier is
        # forbidden for memory issues
        for key in classifiers:
            if SPARSE in available_classifiers[key].get_properties()['input']:
                if 'densifier' in preprocessors:
                    while True:
                        try:
                            cs.add_forbidden_clause(
                                ForbiddenAndConjunction(
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter(
                                            'classifier:__choice__'), key),
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter(
                                            'preprocessor:__choice__'), 'densifier')
                                ))
                            # Success
                            break
                        except ValueError:
                            # Change the default and try again
                            try:
                                default = possible_default_classifier.pop()
                            except IndexError:
                                raise ValueError("Cannot find a legal default configuration.")
                            cs.get_hyperparameter(
                                'classifier:__choice__').default = default

        # which would take too long
        # Combinations of non-linear models with feature learning:
        classifiers_ = ["adaboost", "decision_tree", "extra_trees",
                        "gradient_boosting", "k_nearest_neighbors",
                        "libsvm_svc", "random_forest", "gaussian_nb",
                        "decision_tree", "xgradient_boosting"]
        feature_learning = ["kitchen_sinks", "nystroem_sampler"]

        for c, f in product(classifiers_, feature_learning):
            if c not in classifiers:
                continue
            if f not in preprocessors:
                continue
            while True:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "classifier:__choice__"), c),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "preprocessor:__choice__"), f)))
                    break
                except KeyError:
                    break
                except ValueError as e:
                    # Change the default and try again
                    try:
                        default = possible_default_classifier.pop()
                    except IndexError:
                        raise ValueError(
                            "Cannot find a legal default configuration.")
                    cs.get_hyperparameter(
                        'classifier:__choice__').default = default

        # Won't work
        # Multinomial NB etc don't use with features learning, pca etc
        classifiers_ = ["multinomial_nb"]
        preproc_with_negative_X = ["kitchen_sinks", "pca", "truncatedSVD",
                                   "fast_ica", "kernel_pca", "nystroem_sampler"]

        for c, f in product(classifiers_, preproc_with_negative_X):
            if c not in classifiers:
                continue
            if f not in preprocessors:
                continue
            while True:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "preprocessor:__choice__"), f),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "classifier:__choice__"), c)))
                    break
                except KeyError:
                    break
                except ValueError:
                    # Change the default and try again
                    try:
                        default = possible_default_classifier.pop()
                    except IndexError:
                        raise ValueError(
                            "Cannot find a legal default configuration.")
                    cs.get_hyperparameter(
                        'classifier:__choice__').default = default

        return cs

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
              data_preprocessing_components._preprocessors['rescaling']],
             ["balancing",
              data_preprocessing_components._preprocessors['balancing']]])

        # Add the preprocessing component
        steps.append(['preprocessor',
                      feature_preprocessing_components.FeaturePreprocessorChoice])

        # Add the classification component
        steps.append(['classifier',
                      classification_components.ClassifierChoice])
        return steps

    def _get_estimator_hyperparameter_name(self):
        return "classifier"

