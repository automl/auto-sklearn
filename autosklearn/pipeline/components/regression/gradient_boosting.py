import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant, \
    UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS
from autosklearn.util.common import check_none


class GradientBoosting(AutoSklearnRegressionAlgorithm):
    def __init__(self, loss, learning_rate, max_iter, min_samples_leaf, max_depth,
                 max_leaf_nodes, max_bins, l2_regularization, early_stop, tol, scoring,
                 n_iter_no_change=0, validation_fraction=None, random_state=None,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.early_stop = early_stop
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

    def fit(self, X, y):
        import sklearn.ensemble
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa

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
        self.tol = float(self.tol)
        if check_none(self.scoring):
            self.scoring = None
        if self.early_stop == "off":
            self.n_iter_no_change = 0
            self.validation_fraction = None
        elif self.early_stop == "train":
            self.n_iter_no_change = int(self.n_iter_no_change)
            self.validation_fraction = None
        elif self.early_stop == "valid":
            self.n_iter_no_change = int(self.n_iter_no_change)
            self.validation_fraction = float(self.validation_fraction)
        else:
            raise ValueError("early_stop should be either off, train or valid")
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
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        self.estimator.fit(X, y)
        return self

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
                'handles_multioutput': False,
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
            "max_iter", 32, 512, default_value=100)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True)
        max_depth = UnParametrizedHyperparameter(
            name="max_depth", value="None")
        max_leaf_nodes = UniformIntegerHyperparameter(
            name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True)
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization", lower=1E-10, upper=1, default_value=1E-10, log=True)
        early_stop = CategoricalHyperparameter(
            name="early_stop", choices=["off", "train", "valid"], default_value="off")
        tol = UnParametrizedHyperparameter(
            name="tol", value=1e-7)
        scoring = UnParametrizedHyperparameter(
            name="scoring", value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(
            name="n_iter_no_change", lower=1, upper=20, default_value=10)
        validation_fraction = UniformFloatHyperparameter(
            name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1)

        cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf,
                                max_depth, max_leaf_nodes, max_bins, l2_regularization,
                                early_stop, tol, scoring, n_iter_no_change,
                                validation_fraction])

        n_iter_no_change_cond = InCondition(
            n_iter_no_change, early_stop, ["valid", "train"])
        validation_fraction_cond = EqualsCondition(
            validation_fraction, early_stop, "valid")

        cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])

        return cs
