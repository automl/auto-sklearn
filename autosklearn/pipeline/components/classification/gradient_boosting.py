import numpy as np
import sklearn.ensemble
from sklearn.experimental import enable_hist_gradient_boosting

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_none


class GradientBoostingClassifier(AutoSklearnClassificationAlgorithm):    
    def __init__(self, loss, learning_rate, max_iter, min_samples_leaf, max_depth, 
                 max_leaf_nodes, max_bins, l2_regularization, random_state=None, verbose=0):
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
        self.fully_fit_ = False

    def fit(self, X, Y):
        import sklearn.ensemble
        import sklearn.tree

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

        estimator = sklearn.ensemble.HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_bins = self.max_bins,
            l2_regularization=self.l2_regularization,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        estimator.fit(X, Y)
        self.estimator = estimator
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not (self.estimator.n_iter_ < self.max_iter)

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
        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        max_iter = UniformIntegerHyperparameter(
            "max_iter", 50, 500, default_value=100)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=2, upper=10, default_value=3)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        max_bins = Constant("max_bins", 256)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization", lower=0., upper=1., default_value=0., log=False)
        
        cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf,
                                max_depth, max_leaf_nodes, max_bins, l2_regularization])
        
        return cs

