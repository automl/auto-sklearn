"""
================================================
Extending Auto-Sklearn with Regression Component
================================================

The following example demonstrates how to create a new regression
component for using in auto-sklearn.
"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

import sklearn.metrics
import autosklearn.regression
import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, \
    SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


############################################################################
# Implement kernel ridge regression component for auto-sklearn
# ============================================================

class KernelRidgeRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self, alpha, kernel, gamma, degree, coef0, random_state=None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.alpha = float(self.alpha)
        self.gamma = float(self.gamma)
        self.degree = int(self.degree)
        self.coef0 = float(self.coef0)

        import sklearn.kernel_ridge
        self.estimator = sklearn.kernel_ridge.KernelRidge(alpha=self.alpha,
                                                          kernel=self.kernel,
                                                          gamma=self.gamma,
                                                          degree=self.degree,
                                                          coef0=self.coef0,
                                                          )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KRR',
                'name': 'Kernel Ridge Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            name='alpha', lower=10 ** -5, upper=1, log=True, default_value=1.0)
        kernel = CategoricalHyperparameter(
            name='kernel',
            # We restrict ourselves to two possible kernels for this example
            choices=['polynomial', 'rbf'],
            default_value='polynomial'
        )
        gamma = UniformFloatHyperparameter(
            name='gamma', lower=0.00001, upper=1, default_value=0.1, log=True
        )
        degree = UniformIntegerHyperparameter(
            name='degree', lower=2, upper=5, default_value=3
        )
        coef0 = UniformFloatHyperparameter(
            name='coef0', lower=1e-2, upper=1e2, log=True, default_value=1,
        )
        cs.add_hyperparameters([alpha, kernel, gamma, degree, coef0])
        degree_condition = EqualsCondition(degree, kernel, 'polynomial')
        coef0_condition = EqualsCondition(coef0, kernel, 'polynomial')
        cs.add_conditions([degree_condition, coef0_condition])
        return cs


# Add KRR component to auto-sklearn.
autosklearn.pipeline.components.regression.add_regressor(KernelRidgeRegression)
cs = KernelRidgeRegression.get_hyperparameter_search_space()
print(cs)


if __name__ == "__main__":

    ############################################################################
    # Generate data
    # =============

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ############################################################################
    # Fit the model using KRR
    # =======================

    reg = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        include_estimators=['KernelRidgeRegression'],
        # Bellow two flags are provided to speed up calculations
        # Not recommended for a real implementation
        initial_configurations_via_metalearning=0,
        smac_scenario_args={'runcount_limit': 5},
    )
    reg.fit(X_train, y_train)

    ############################################################################
    # Print prediction score and statistics
    # =====================================
    y_pred = reg.predict(X_test)
    print("r2 score: ", sklearn.metrics.r2_score(y_pred, y_test))
    print(reg.show_models())
