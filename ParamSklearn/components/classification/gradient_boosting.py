import numpy as np
import sklearn.ensemble

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, PREDICTIONS


class GradientBoostingClassifier(ParamSklearnClassificationAlgorithm):

    def __init__(self, learning_rate, n_estimators, subsample,
                 min_samples_split, min_samples_leaf, max_features, max_depth,
                 max_leaf_nodes_or_max_depth="max_depth",
                 max_leaf_nodes=None, loss='deviance',
                 init=None, random_state=None, verbose=0, estimator_increment=10):

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

        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.estimator_increment = int(estimator_increment)
        self.subsample = float(subsample)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        if max_features in ("sqrt", "log2", "auto"):
            raise ValueError("'max_features' should be a float: %s" %
                             max_features)
        self.max_features = float(max_features)

        self.loss = loss
        self.init = init
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, y, sample_weight=None, refit=False):
        if self.estimator is None or refit:
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight,
                               refit=refit)

        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1, sample_weight=sample_weight)
        return self

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        if refit:
            self.estimator = None

        if self.estimator is None:
            num_features = X.shape[1]
            max_features = int(float(self.max_features) * (np.log(num_features) + 1))
            # Use at most half of the features
            max_features = max(1, min(int(X.shape[1] / 2), max_features))
            self.estimator = sklearn.ensemble.GradientBoostingClassifier(
                learning_rate=self.learning_rate,
                n_estimators=0,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                loss=self.loss,
                max_depth=self.max_depth,
                warm_start=True,
                init=self.init,
                random_state=self.random_state,
                verbose=self.verbose
            )

        tmp = self.estimator  # TODO copy ?
        tmp.n_estimators += n_iter
        tmp.fit(X, y, sample_weight=sample_weight)
        self.estimator = tmp

        return self


    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, ),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.0001, upper=1, default=0.1, log=True)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default=1.0, log=False)

        # Unparametrized
        #max_leaf_nodes_or_max_depth = UnParametrizedHyperparameter(
        #    name="max_leaf_nodes_or_max_depth", value="max_depth")
            # CategoricalHyperparameter("max_leaf_nodes_or_max_depth",
            # choices=["max_leaf_nodes", "max_depth"], default="max_depth")

        max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes",
                                                      value="None")


        # Copied from random_forest.py
        #n_estimators = UniformIntegerHyperparameter(
        #    name="n_estimators", lower=10, upper=100, default=10, log=False)
        n_estimators = Constant("n_estimators", 100)
        #max_features = UniformFloatHyperparameter(
        #    name="max_features", lower=0.01, upper=0.5, default=0.1)
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
        cs.add_hyperparameter(learning_rate)
        cs.add_hyperparameter(max_features)
        #cs.add_hyperparameter(max_leaf_nodes_or_max_depth)
        #cs.add_hyperparameter(max_leaf_nodes)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(subsample)

        # Conditions
        #cond_max_leaf_nodes_or_max_depth = \
        #    EqualsCondition(child=max_leaf_nodes,
        #                    parent=max_leaf_nodes_or_max_depth,
        #                    value="max_leaf_nodes")

        #cond2_max_leaf_nodes_or_max_depth = \
        #    EqualsCondition(child=max_depth,
        #                    parent=max_leaf_nodes_or_max_depth,
        #                    value="max_depth")

        #cs.add_condition(cond_max_leaf_nodes_or_max_depth)
        #cs.add_condition(cond2_max_leaf_nodes_or_max_depth)
        return cs

