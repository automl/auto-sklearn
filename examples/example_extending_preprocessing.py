"""
===============================================
Extending Auto-sklearn
===============================================

In order to include new machine learning algorithms in auto-sklearn's
optimization process, users can implement a wrapper class for the algorithm
and register it to auto-sklearn. The example code below demonstrates how
to implement custom regressor and preprocessor (Lasso and polynomial processing from sklearn, respectively),
register it to auto-sklearn, and use them for the given task.
A detailed walkthrough of extending auto-sklearn can be found `here <https://automl.github.io/auto-sklearn/stable/extending.html>`_.

"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import *
from ConfigSpace.conditions import EqualsCondition, InCondition

from autosklearn.pipeline.components.base import \
    AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool


# Custom Regression algorithm added to auto-sklearn (Lasso from sklearn).
class MyRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, alpha, fit_intercept, tol, positive, random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        #self.normalize = normalize
        self.tol = tol
        self.positive = positive

        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.linear_model

        self.alpha = float(self.alpha)
        self.fit_intercept = check_for_bool(self.fit_intercept)
        self.normalize = check_for_bool(self.normalize)
        self.tol = float(self.tol)
        self.positive = check_for_bool(self.positive)

        self.estimator = sklearn.linear_model.\
            Lasso(alpha=self.alpha,
                  fit_intercept=self.fit_intercept,
                  tol=self.tol,
                  positive=self.positive,
                  n_iter=300)

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MyRegressor',
                'name': 'MyRegressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0, upper=10, default_value=1)
        fit_intercept = CategoricalHyperparameter(
            name="fit_intercept", choices=[True, False], default_value=True)
        normalize = CategoricalHyperparameter(
            name="normalize", choices=[True, False], default_value=False)
        tol = UniformFloatHyperparameter(
            name="tol", lower=10 ** -5, upper=10 ** -1,
            default_value=10 ** -3, log=True)
        positive = CategoricalHyperparameter(
            name="positive", choices=[True, False], default_value=False)

        cs.add_hyperparameters([alpha, fit_intercept, tol, positive])

        return cs


# Custom wrapper class for using Sklearn's polynomial feature preprocessing
# function.
class MyPreprocessor(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, degree, interaction_only, include_bias, random_state=None):
        # Define hyperparameters to be tuned here.
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        # wrapper function for the fit method of Sklearn's polynomial
        # preprocessing function.
        import sklearn.preprocessing
        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(degree=self.degree,
                                                                     interaction_only=self.interaction_only,
                                                                     include_bias=self.include_bias)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        # wrapper function for the transform method of sklearn's polynomial
        # preprocessing function. It is also possible to implement
        # a preprocessing algorithm directly in this function, provided that
        # it behaves in the way compatible with that from sklearn.
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MyPreprocessor',
                'name': 'MyPreprocessor',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # For each hyperparameter, its type (categorical, integer, float, etc.),
        # range and the default value must be specified here.
        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=2)
        interaction_only = CategoricalHyperparameter(
            name="interaction_only", choices=["False", "True"], default_value="False")
        include_bias = CategoricalHyperparameter(
            name="include_bias", choices=["True", "False"], default_value="True")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs


def main():
    # Include the custom preprocessor class to auto-sklearn.
    import autosklearn.pipeline.components.regression
    import autosklearn.pipeline.components.feature_preprocessing
    autosklearn.pipeline.components.regression.add_regressor(MyRegressor)
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(MyPreprocessor)

    # Import toy data from sklearn and apply train_test_split.
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Run auto-sklearn regression with the custom preprocessor.
    import autosklearn.regression
    import autosklearn.metrics
    reg = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=30,
                                                      per_run_time_limit=10,
                                                      include_estimators=["MyRegressor"],
                                                      include_preprocessors=["MyPreprocessor"])
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    scorer = autosklearn.metrics.r2
    print("Test score: ", scorer(y_pred, y_test))
    print(reg.show_models())
    print(reg.sprint_statistics())


if __name__ == "__main__":
    main()
