from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UnParametrizedHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS
from autosklearn.util.common import check_for_bool


class ARDRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2,
                 threshold_lambda, fit_intercept, random_state=None):
        self.random_state = random_state
        self.estimator = None

        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.threshold_lambda = threshold_lambda
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        import sklearn.linear_model

        self.n_iter = int(self.n_iter)
        self.tol = float(self.tol)
        self.alpha_1 = float(self.alpha_1)
        self.alpha_2 = float(self.alpha_2)
        self.lambda_1 = float(self.lambda_1)
        self.lambda_2 = float(self.lambda_2)
        self.threshold_lambda = float(self.threshold_lambda)
        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.estimator = sklearn.linear_model.\
            ARDRegression(n_iter=self.n_iter,
                          tol=self.tol,
                          alpha_1=self.alpha_1,
                          alpha_2=self.alpha_2,
                          lambda_1=self.lambda_1,
                          lambda_2=self.lambda_2,
                          compute_score=False,
                          threshold_lambda=self.threshold_lambda,
                          fit_intercept=True,
                          normalize=False,
                          copy_X=False,
                          verbose=False)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'ARD',
                'name': 'ARD Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'prefers_data_normalized': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_iter = UnParametrizedHyperparameter("n_iter", value=300)
        tol = UniformFloatHyperparameter("tol", 10 ** -5, 10 ** -1,
                                         default_value=10 ** -3, log=True)
        alpha_1 = UniformFloatHyperparameter(name="alpha_1", lower=10 ** -10,
                                             upper=10 ** -3, default_value=10 ** -6)
        alpha_2 = UniformFloatHyperparameter(name="alpha_2", log=True,
                                             lower=10 ** -10, upper=10 ** -3,
                                             default_value=10 ** -6)
        lambda_1 = UniformFloatHyperparameter(name="lambda_1", log=True,
                                              lower=10 ** -10, upper=10 ** -3,
                                              default_value=10 ** -6)
        lambda_2 = UniformFloatHyperparameter(name="lambda_2", log=True,
                                              lower=10 ** -10, upper=10 ** -3,
                                              default_value=10 ** -6)
        threshold_lambda = UniformFloatHyperparameter(name="threshold_lambda",
                                                      log=True,
                                                      lower=10 ** 3,
                                                      upper=10 ** 5,
                                                      default_value=10 ** 4)
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")

        cs.add_hyperparameters([n_iter, tol, alpha_1, alpha_2, lambda_1,
                                lambda_2, threshold_lambda, fit_intercept])

        return cs
