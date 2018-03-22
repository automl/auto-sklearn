import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import InCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
    IterativeComponent,
)
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_none, check_for_bool

class GradientBoosting(
    IterativeComponent,
    AutoSklearnRegressionAlgorithm,
):
    def __init__(self, loss, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_depth, max_features,
                 max_leaf_nodes, min_impurity_decrease, alpha=None, init=None,
                 random_state=None, verbose=0):
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
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha = alpha
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        import sklearn.ensemble

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
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            self.max_features = float(self.max_features)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.min_impurity_decrease = float(self.min_impurity_decrease)
            if not check_none(self.alpha):
                self.alpha = float(self.alpha)
            self.verbose = int(self.verbose)

            self.estimator = sklearn.ensemble.GradientBoostingRegressor(
                loss=self.loss,
                learning_rate=self.learning_rate,
                n_estimators=n_iter,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                random_state=self.random_state,
                verbose=self.verbose,
                warm_start=True,
            )
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y, sample_weight=sample_weight)

        return self


    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'prefers_data_normalized': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        loss = CategoricalHyperparameter(
            "loss", ["ls", "lad", "huber", "quantile"], default_value="ls")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", 50, 500, default_value=100)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=3)
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default_value=2, log=False)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1, log=False)
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            name='min_impurity_decrease', value=0.0)
        alpha = UniformFloatHyperparameter(
            "alpha", lower=0.75, upper=0.99, default_value=0.9)

        cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                                min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, subsample, max_features,
                                max_leaf_nodes, min_impurity_decrease, alpha])

        cs.add_condition(InCondition(alpha, loss, ['huber', 'quantile']))
        return cs
