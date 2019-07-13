import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import InCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
#    IterativeComponent,
)
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_none, check_for_bool

class GradientBoosting(
    #IterativeComponent,
    AutoSklearnRegressionAlgorithm,
):
    #def __init__(self, loss, learning_rate, n_estimators, subsample,
    #             min_samples_split, min_samples_leaf,
    #             min_weight_fraction_leaf, max_depth, max_features,
    #             max_leaf_nodes, min_impurity_decrease, alpha=None, init=None,
    #             random_state=None, verbose=0):
    def __init__(self, loss, learning_rate, max_iter, min_samples_leaf, 
                max_depth, max_leaf_nodes, max_bins, l2_regularization,
                random_state=None, verbose=0):        
        self.loss = loss
        self.learning_rate = learning_rate
        #self.n_estimators = n_estimators
        self.max_iter = max_iter
        #self.subsample = subsample
        #self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        #self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        #self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        #self.min_impurity_decrease = min_impurity_decrease
        #self.alpha = alpha
        #self.init = init
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

    #def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
    def fit(self, X, y):
        import sklearn.ensemble
        from sklearn.experimental import enable_hist_gradient_boosting 

        # Special fix for gradient boosting!
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=X.dtype)

        self.learning_rate = float(self.learning_rate)
        self.max_iter = int(self.max_iter)
        #self.subsample = float(self.subsample)
        #self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        #self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        #self.max_features = float(self.max_features)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        #self.min_impurity_decrease = float(self.min_impurity_decrease)
        #if not check_none(self.alpha):
        #    self.alpha = float(self.alpha)
        self.max_bins = int(self.max_bins)
        self.l2_regularization = float(self.l2_regularization)
        self.verbose = int(self.verbose)

        self.estimator = sklearn.ensemble.HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            #n_estimators=n_iter,
            max_iter=self.max_iter,
            #subsample=self.subsample,
            #min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            #min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            #max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            #min_impurity_decrease=self.min_impurity_decrease,
            #init=self.init,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            verbose=self.verbose,
            #warm_start=True,
        )

        #self.estimator.fit(X, y, sample_weight=sample_weight)
        self.estimator.fit(X, y)

        return self


    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        #return not len(self.estimator.estimators_) < self.n_estimators
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
        #n_estimators = UniformIntegerHyperparameter(
        #    "n_estimators", 50, 500, default_value=100)
        max_iter = UniformIntegerHyperparameter(
            "max_iter", 50, 500, default_value=100)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=3)
        #min_samples_split = UniformIntegerHyperparameter(
        #    name="min_samples_split", lower=2, upper=20, default_value=2, log=False)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1, log=False)
        #min_weight_fraction_leaf = UnParametrizedHyperparameter(
        #    "min_weight_fraction_leaf", 0.)
        #subsample = UniformFloatHyperparameter(
        #    name="subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
        #max_features = UniformFloatHyperparameter(
        #    "max_features", 0.1, 1.0, default_value=1)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        #min_impurity_decrease = UnParametrizedHyperparameter(
        #    name='min_impurity_decrease', value=0.0)
        #alpha = UniformFloatHyperparameter(
        #    "alpha", lower=0.75, upper=0.99, default_value=0.9)
        max_bins = UniformIntegerHyperparameter(
            name="max_bins", lower=2, upper=256, default_value=256, log=False)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization", lower=0., upper=1., default_value=0., log=False)

        cs.add_hyperparameters([loss, learning_rate, max_iter, max_depth, min_samples_leaf,
                                max_leaf_nodes, max_bins, l2_regularization])

        #cs.add_condition(InCondition(alpha, loss, ['huber', 'quantile']))
        return cs
