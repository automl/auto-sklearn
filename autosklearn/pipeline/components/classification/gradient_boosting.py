import numpy as np
import sklearn.ensemble

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *


class GradientBoostingClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, max_features,
                 max_leaf_nodes, init=None, random_state=None, verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    def fit(self, X, y, sample_weight=None, refit=False):
        self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight,
                           refit=True)
        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight)
        return self

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.learning_rate = float(self.learning_rate)
            self.n_estimators = int(self.n_estimators)
            self.subsample = float(self.subsample)
            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            if self.max_depth == "None":
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            num_features = X.shape[1]
            max_features = int(
                float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            if self.max_leaf_nodes == "None":
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.verbose = int(self.verbose)

            self.estimator = sklearn.ensemble.GradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=n_iter,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                max_features=max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                init=self.init,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )

        else:
            self.estimator.n_estimators += n_iter

        self.estimator.fit(X, y, sample_weight=sample_weight)

        # Apparently this if is necessary
        if self.estimator.n_estimators >= self.n_estimators:
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        loss = Constant("loss", "deviance")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", 50, 500, default=100)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=3)
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default=2, log=False)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default=1, log=False)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        subsample = UniformFloatHyperparameter(
                name="subsample", lower=0.01, upper=1.0, default=1.0, log=False)
        max_features = UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                                min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, subsample,
                                max_features, max_leaf_nodes])

        return cs

