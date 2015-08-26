import numpy as np
import sklearn.ensemble
import sklearn.tree

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

from ParamSklearn.components.base import ParamSklearnRegressionAlgorithm
from ParamSklearn.constants import *


class AdaboostRegressor(ParamSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, learning_rate, loss, max_depth,
                 random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, Y):
        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeClassifier(
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
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS, ),
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

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

