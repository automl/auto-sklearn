import numpy as np
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter

from ParamSklearn.components.base import \
    ParamSklearnClassificationAlgorithm
from ParamSklearn.constants import *
from ParamSklearn.implementations.util import softmax


class PassiveAggressive(ParamSklearnClassificationAlgorithm):
    def __init__(self, C, fit_intercept, n_iter, loss, random_state=None):
        self.C = float(C)
        self.fit_intercept = fit_intercept == 'True'
        self.n_iter = int(n_iter)
        self.loss = loss
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1)

        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.estimator = PassiveAggressiveClassifier(
                C=self.C, fit_intercept=self.fit_intercept, n_iter=1,
                loss=self.loss, shuffle=True, random_state=self.random_state,
                warm_start=True)
            self.classes_ = np.unique(y.astype(int))

        self.estimator.n_iter += n_iter
        self.estimator.fit(X, y)

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not self.estimator.n_iter < self.n_iter

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.decision_function(X)
        return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PassiveAggressive Classifier',
                'name': 'Passive Aggressive Stochastic Gradient Descent '
                        'Classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        loss = CategoricalHyperparameter("loss",
                                         ["hinge", "squared_hinge"],
                                         default="hinge")
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        n_iter = UniformIntegerHyperparameter("n_iter", 5, 1000, default=20)
        C = UniformFloatHyperparameter("C", 1e-5, 10, 1, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(loss)
        cs.add_hyperparameter(fit_intercept)
        cs.add_hyperparameter(n_iter)
        cs.add_hyperparameter(C)
        return cs
