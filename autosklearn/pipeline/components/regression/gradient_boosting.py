import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import InCondition

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_none, check_for_bool

class GradientBoosting(AutoSklearnRegressionAlgorithm):
    def __init__(self, loss, learning_rate, max_iter, min_samples_leaf, 
                max_depth, max_leaf_nodes, max_bins, l2_regularization,
                random_state=None, verbose=0):        
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

    def fit(self, X, y):
        import sklearn.ensemble
        from sklearn.experimental import enable_hist_gradient_boosting 

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)

        self.learning_rate = float(self.learning_rate)
        self.max_iter = int(self.max_iter)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.max_bins = int(self.max_bins)
        self.l2_regularization = float(self.l2_regularization)
        self.verbose = int(self.verbose)

        self.estimator = sklearn.ensemble.HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.estimator.fit(X, y)

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not (self.estimator.n_iter_ < self.max_iter)

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
            "loss", ["least_squares"], default_value="least_squares")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        max_iter = UniformIntegerHyperparameter(
            "max_iter", 50, 500, default_value=100)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1, log=False)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=3)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        max_bins = Constant("max_bins", 256)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization", lower=0., upper=1., default_value=0., log=False)

        cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf,
                                max_depth, max_leaf_nodes, max_bins, 
                                l2_regularization])

        return cs
