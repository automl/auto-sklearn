import numpy as np
import sklearn.ensemble

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.conditions import InCondition
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ..regression_base import AutoSklearnRegressionAlgorithm


class GradientBoosting(AutoSklearnRegressionAlgorithm):
    def __init__(self,
                 loss, learning_rate, subsample, min_samples_split,
                 min_samples_leaf, max_depth, max_features, alpha=0.9,
                 max_leaf_nodes=None, estimator_increment=10,
                 max_leaf_nodes_or_max_depth="max_depth",
                 n_estimators=100, init=None, random_state=None, verbose=0):

        self.max_leaf_nodes_or_max_depth = str(max_leaf_nodes_or_max_depth)

        if self.max_leaf_nodes_or_max_depth == "max_depth":
            if max_depth == 'None':
                self.max_depth = None
            else:
                self.max_depth = int(max_depth)
            self.max_leaf_nodes = None
        elif self.max_leaf_nodes_or_max_depth == "max_leaf_nodes":
            self.max_depth = None
            if max_leaf_nodes == 'None':
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(max_leaf_nodes)
        else:
            raise ValueError("max_leaf_nodes_or_max_depth sould be in "
                             "('max_leaf_nodes', 'max_depth'): %s" %
                             self.max_leaf_nodes_or_max_depth)

        if loss in ("ls", "lad", "huber", "quantile"):
            self.loss = loss
        else:
            raise ValueError("'loss' should be in ('ls', 'lad', 'huber', "
                             "'quantile'), but is %s" % str(loss))
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.min_samples_split = int(float(min_samples_split))
        self.min_samples_leaf = int(float(min_samples_leaf))
        self.max_depth = int(float(max_depth))

        if self.loss in ('huber', 'quantile'):
            self.alpha = float(alpha)
        else:
            self.alpha = 0.9  # default value

        self.n_estimators = n_estimators

        self.estimator_increment = int(estimator_increment)
        self.init = init

        # We handle this later
        self.max_features = float(max_features)

        # Defaults
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, Y):
        num_features = X.shape[1]
        max_features = int(float(self.max_features) * (np.log(num_features) + 1))
        # Use at most half of the features
        max_features = max(1, min(int(X.shape[1] / 2), max_features))

        self.estimator = sklearn.ensemble.GradientBoostingRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=0,
            subsample=self.subsample,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            init=self.init,
            max_features=max_features,
            alpha=self.alpha,
            warm_start=True,
            random_state=self.random_state,
            verbose=self.verbose
        )
        # JTS TODO: I think we might have to copy here if we want self.estimator
        #           to always be consistent on sigabort
        while len(self.estimator.estimators_) < self.n_estimators:
            tmp = self.estimator # TODO I think we need to copy here!
            tmp.n_estimators += self.estimator_increment
            tmp.fit(X, Y)
            self.estimator = tmp
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Regressor',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'is_deterministic': True,
                'handles_sparse': False,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        loss = CategoricalHyperparameter(
            name="loss", choices=["ls", "lad"], default='ls') #, "huber", "quantile"], default='ls')

        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.0001, upper=1, default=0.1, log=True)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default=1.0, log=False)

        n_estimators = Constant("n_estimators", 100)

        max_features = UniformFloatHyperparameter(
            "max_features", 0.5, 5, default=1)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=3)
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default=2, log=False)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default=1, log=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(loss)
        cs.add_hyperparameter(learning_rate)
        cs.add_hyperparameter(max_features)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(subsample)
        return cs