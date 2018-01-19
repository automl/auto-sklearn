import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class ExtraTreesRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_impurity_decrease, oob_score=False, n_jobs=1,
                 random_state=None, verbose=0):

        self.n_estimators = int(n_estimators)
        self.estimator_increment = 10
        if criterion not in ("mse", "friedman_mse", "mae"):
            raise ValueError("'criterion' is not in ('mse', 'friedman_mse', "
                             "'mae): %s" % criterion)
        self.criterion = criterion

        if max_depth == "None" or max_depth is None:
            self.max_depth = None
        else:
            self.max_depth = int(max_depth)
        if max_leaf_nodes == "None" or max_leaf_nodes is None:
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(max_leaf_nodes)

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)

        self.max_features = float(max_features)
        self.min_impurity_decrease = float(min_impurity_decrease)

        if bootstrap == "True":
            self.bootstrap = True
        elif bootstrap == "False":
            self.bootstrap = False

        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, y, refit=False):
        n_iter = 2
        self.iterative_fit(X, y, n_iter=n_iter, refit=True)
        while not self.configuration_fully_fitted():
            n_iter *= 2
            self.iterative_fit(X, y, n_iter=n_iter)

        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        from sklearn.ensemble import ExtraTreesRegressor as ETR

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.estimator = ETR(n_estimators=n_iter,
                                 criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 bootstrap=self.bootstrap,
                                 max_features=self.max_features,
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_impurity_decrease=self.min_impurity_decrease,
                                 oob_score=self.oob_score,
                                 n_jobs=self.n_jobs,
                                 verbose=self.verbose,
                                 random_state=self.random_state,
                                 warm_start=True)
        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(self.estimator.n_estimators,
                                              self.n_estimators)

        self.estimator.fit(X, y,)

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
    def get_properties(dataset_properties=None):
        return {'shortname': 'ET',
                'name': 'Extra Trees Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        n_estimators = Constant("n_estimators", 100)
        criterion = CategoricalHyperparameter("criterion",
                                              ['mse', 'friedman_mse', 'mae'])
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1)

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_impurity_decrease = UnParametrizedHyperparameter(
            'min_impurity_decrease', 0.0
        )

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False")

        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, max_leaf_nodes, min_samples_split,
                                min_samples_leaf, min_impurity_decrease,
                                bootstrap])

        return cs
