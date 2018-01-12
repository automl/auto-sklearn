from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none

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

        self.dual = check_for_bool(self.dual)
        self.fit_intercept = check_for_bool(self.fit_intercept)
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
        C = UniformFloatHyperparameter(
            "C", 0.03125, 32768, log=True, default_value=1.0)
        loss = CategoricalHyperparameter(
            "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"],
            default_value="squared_epsilon_insensitive")
        # Random Guess
        epsilon = UniformFloatHyperparameter(
            name="epsilon", lower=0.001, upper=1, default_value=0.1, log=True)
        dual = Constant("dual", "False")
        # These are set ad-hoc
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        fit_intercept =Constant("fit_intercept", "True")
        intercept_scaling = Constant("intercept_scaling", 1)

        cs.add_hyperparameters([C, loss, epsilon, dual, tol, fit_intercept,
                                intercept_scaling])

        dual_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(loss, "epsilon_insensitive")
        )
        cs.add_forbidden_clause(dual_and_loss)

        return cs
