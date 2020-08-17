import copy
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor
from autosklearn.pipeline.components import classification as \
    classification_components
from autosklearn.pipeline.components.data_preprocessing.balancing.balancing import \
    Balancing
from autosklearn.pipeline.components import feature_preprocessing as \
    feature_preprocessing_components
from autosklearn.pipeline.base import BasePipeline
from autosklearn.pipeline.constants import SPARSE


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
    config : ConfigSpace.configuration_space.Configuration
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

    def __init__(self, config=None, steps=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None):
        self._output_dtype = np.int32
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params)

    def fit_transformer(self, X, y, fit_params=None):

        if fit_params is None:
            fit_params = {}

        if self.config['balancing:strategy'] == 'weighting':
            balancing = Balancing(strategy='weighting')
            _init_params, _fit_params = balancing.get_weights(
                y, self.config['classifier:__choice__'],
                self.config['feature_preprocessor:__choice__'],
                {}, {})
            _init_params.update(self.init_params)
            self.set_hyperparameters(configuration=self.config,
                                     init_params=_init_params)

            if _fit_params is not None:
                fit_params.update(_fit_params)

        X, fit_params = super().fit_transformer(
            X, y, fit_params=fit_params)

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
            return super().predict_proba(X)

        else:
            if not isinstance(batch_size, int):
                raise ValueError("Argument 'batch_size' must be of type int, "
                                 "but is '%s'" % type(batch_size))
            if batch_size <= 0:
                raise ValueError("Argument 'batch_size' must be positive, "
                                 "but is %d" % batch_size)

            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0:2].copy())

                y = np.zeros((X.shape[0], target.shape[1]),
                             dtype=np.float32)

                for k in range(max(1, int(np.ceil(float(X.shape[0]) / batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    pred_prob = self.predict_proba(X[batch_from:batch_to], batch_size=None)
                    y[batch_from:batch_to] = pred_prob.astype(np.float32)

                return y

    def _get_hyperparameter_search_space(self, include=None, exclude=None,
                                         dataset_properties=None):
        """Create the hyperparameter configuration space.

        Parameters
        ----------
        include : dict (optional, default=None)

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()
        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'classification'
        if dataset_properties['target_type'] != 'classification':
            dataset_properties['target_type'] = 'classification'

        if 'sparse' not in dataset_properties:
            # This dataset is probably dense
            dataset_properties['sparse'] = False

        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        classifiers = cs.get_hyperparameter('classifier:__choice__').choices
        preprocessors = cs.get_hyperparameter('feature_preprocessor:__choice__').choices
        available_classifiers = self._final_estimator.get_available_components(
            dataset_properties)

        possible_default_classifier = copy.copy(list(
            available_classifiers.keys()))
        default = cs.get_hyperparameter('classifier:__choice__').default_value
        del possible_default_classifier[possible_default_classifier.index(default)]

        # A classifier which can handle sparse data after the densifier is
        # forbidden for memory issues
        for key in classifiers:
            if SPARSE in available_classifiers[key].get_properties()['input']:
                if 'densifier' in preprocessors:
                    while True:
                        try:
                            forb_cls = ForbiddenEqualsClause(
                                cs.get_hyperparameter('classifier:__choice__'), key)
                            forb_fpp = ForbiddenEqualsClause(cs.get_hyperparameter(
                                'feature_preprocessor:__choice__'), 'densifier')
                            cs.add_forbidden_clause(
                                ForbiddenAndConjunction(forb_cls, forb_fpp))
                            # Success
                            break
                        except ValueError:
                            # Change the default and try again
                            try:
                                default = possible_default_classifier.pop()
                            except IndexError:
                                raise ValueError(
                                    "Cannot find a legal default configuration.")
                            cs.get_hyperparameter(
                                'classifier:__choice__').default_value = default

        # which would take too long
        # Combinations of non-linear models with feature learning:
        classifiers_ = ["adaboost", "decision_tree", "extra_trees",
                        "gradient_boosting", "k_nearest_neighbors",
                        "libsvm_svc", "random_forest", "gaussian_nb",
                        "decision_tree"]
        feature_learning = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]

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
                            "feature_preprocessor:__choice__"), f)))
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
                        'classifier:__choice__').default_value = default

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
                            "feature_preprocessor:__choice__"), f),
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
                        'classifier:__choice__').default_value = default

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _get_pipeline_steps(self, dataset_properties):
        steps = []

        default_dataset_properties = {'target_type': 'classification'}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ["data_preprocessing",
                DataPreprocessor(dataset_properties=default_dataset_properties)],
            ["balancing",
                Balancing()],
            ["feature_preprocessor",
                feature_preprocessing_components.FeaturePreprocessorChoice(
                    default_dataset_properties)],
            ['classifier',
                classification_components.ClassifierChoice(
                    default_dataset_properties)]
        ])

        return steps

    def _get_estimator_hyperparameter_name(self):
        return "classifier"
