from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class LibLinear_SVR(AutoSklearnRegressionAlgorithm):
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, loss, epsilon, dual, tol, C, fit_intercept,
                 intercept_scaling, random_state=None):
        self.epsilon = epsilon
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        self.C = float(self.C)
        self.tol = float(self.tol)
        self.epsilon = float(self.epsilon)

        self.dual = self.dual == 'True'
        self.fit_intercept = self.fit_intercept == 'True'
        self.intercept_scaling = float(self.intercept_scaling)

        self.estimator = sklearn.svm.LinearSVR(epsilon=self.epsilon,
                                               loss=self.loss,
                                               dual=self.dual,
                                               tol=self.tol,
                                               C=self.C,
                                               fit_intercept=self.fit_intercept,
                                               intercept_scaling=self.intercept_scaling,
                                               random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Liblinear-SVR',
                'name': 'Liblinear Support Vector Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        C = cs.add_hyperparameter(UniformFloatHyperparameter(
            "C", 0.03125, 32768, log=True, default=1.0))
        loss = cs.add_hyperparameter(CategoricalHyperparameter(
            "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"],
            default="squared_epsilon_insensitive"))
        # Random Guess
        epsilon = cs.add_hyperparameter(UniformFloatHyperparameter(
            name="epsilon", lower=0.001, upper=1, default=0.1, log=True))
        dual = cs.add_hyperparameter(Constant("dual", "False"))
        # These are set ad-hoc
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))
        fit_intercept = cs.add_hyperparameter(Constant("fit_intercept", "True"))
        intercept_scaling = cs.add_hyperparameter(Constant(
            "intercept_scaling", 1))

        dual_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(loss, "epsilon_insensitive")
        )
        cs.add_forbidden_clause(dual_and_loss)

        return cs
