from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition, EqualsCondition

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class SGD(AutoSklearnRegressionAlgorithm):
    def __init__(self, loss, penalty, alpha, fit_intercept, n_iter,
                 learning_rate, l1_ratio=0.15, epsilon=0.1,
                 eta0=0.01, power_t=0.5, average=False, random_state=None):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average

        self.estimator = None
        self.scaler = None

    def fit(self, X, y):
        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1)

        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        from sklearn.linear_model.stochastic_gradient import SGDRegressor
        import sklearn.preprocessing

        if refit:
            self.estimator = None
            self.scaler = None

        if self.estimator is None:
            self._iterations = 0

            self.alpha = float(self.alpha)
            self.fit_intercept = self.fit_intercept == 'True'
            self.n_iter = int(self.n_iter)
            self.l1_ratio = float(
                self.l1_ratio) if self.l1_ratio is not None else 0.15
            self.epsilon = float(
                self.epsilon) if self.epsilon is not None else 0.1
            self.eta0 = float(self.eta0)
            self.power_t = float(
                self.power_t) if self.power_t is not None else 0.25
            self.average = self.average == 'True'
            self.estimator = SGDRegressor(loss=self.loss,
                                          penalty=self.penalty,
                                          alpha=self.alpha,
                                          fit_intercept=self.fit_intercept,
                                          n_iter=self.n_iter,
                                          learning_rate=self.learning_rate,
                                          l1_ratio=self.l1_ratio,
                                          epsilon=self.epsilon,
                                          eta0=self.eta0,
                                          power_t=self.power_t,
                                          shuffle=True,
                                          average=self.average,
                                          random_state=self.random_state)

            self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
            self.scaler.fit(y.reshape((-1, 1)))

        Y_scaled = self.scaler.transform(y.reshape((-1, 1))).ravel()

        self.estimator.n_iter = n_iter
        self._iterations += n_iter
        self.estimator.partial_fit(X, Y_scaled)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not self._iterations < self.n_iter

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        Y_pred = self.estimator.predict(X)
        return self.scaler.inverse_transform(Y_pred)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SGD Regressor',
                'name': 'Stochastic Gradient Descent Regressor',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        loss = cs.add_hyperparameter(CategoricalHyperparameter("loss",
            ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            default="squared_loss"))
        penalty = cs.add_hyperparameter(CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default="l2"))
        alpha = cs.add_hyperparameter(UniformFloatHyperparameter(
            "alpha", 10e-7, 1e-1, log=True, default=0.01))
        l1_ratio = cs.add_hyperparameter(UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1., log=True, default=0.15))
        fit_intercept = cs.add_hyperparameter(UnParametrizedHyperparameter(
            "fit_intercept", "True"))
        n_iter = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "n_iter", 5, 1000, log=True, default=20))
        epsilon = cs.add_hyperparameter(UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default=1e-4, log=True))
        learning_rate = cs.add_hyperparameter(CategoricalHyperparameter(
            "learning_rate", ["optimal", "invscaling", "constant"],
            default="optimal"))
        eta0 = cs.add_hyperparameter(UniformFloatHyperparameter(
            "eta0", 10 ** -7, 0.1, default=0.01))
        power_t = cs.add_hyperparameter(UniformFloatHyperparameter(
            "power_t", 1e-5, 1, default=0.5))
        average = cs.add_hyperparameter(CategoricalHyperparameter(
            "average", ["False", "True"], default="False"))

        # TODO add passive/aggressive here, although not properly documented?
        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = InCondition(epsilon, loss,
            ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
        # eta0 seems to be always active according to the source code; when
        # learning_rate is set to optimial, eta0 is the starting value:
        # https://github.com/scikit-learn/scikit-learn/blob/0.15.X/sklearn/linear_model/sgd_fast.pyx
        # eta0_and_inv = EqualsCondition(eta0, learning_rate, "invscaling")
        #eta0_and_constant = EqualsCondition(eta0, learning_rate, "constant")
        #eta0_condition = OrConjunction(eta0_and_inv, eta0_and_constant)
        power_t_condition = EqualsCondition(power_t, learning_rate,
                                            "invscaling")

        cs.add_condition(elasticnet)
        cs.add_condition(epsilon_condition)
        cs.add_condition(power_t_condition)

        return cs
