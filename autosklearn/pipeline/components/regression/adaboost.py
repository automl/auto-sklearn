import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class AdaboostRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, learning_rate, loss, max_depth,
                 random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.ensemble
        import sklearn.tree

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeRegressor(
            max_depth=self.max_depth)

        self.estimator = sklearn.ensemble.AdaBoostRegressor(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=self.random_state
        )
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'AdaBoost Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS, )}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        # base_estimator = Constant(name="base_estimator", value="None")
        n_estimators = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default=50, log=False))
        learning_rate = cs.add_hyperparameter(UniformFloatHyperparameter(
            name="learning_rate", lower=0.0001, upper=2, default=0.1, log=True))
        loss = cs.add_hyperparameter(CategoricalHyperparameter(
            name="loss", choices=["linear", "square", "exponential"],
            default="linear"))
        max_depth = cs.add_hyperparameter(UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default=1, log=False))
        return cs

