"""
===============================================
Extending Auto-sklearn with Custom Preprocessor
===============================================


explanation goes here.
"""

import autosklearn.pipeline.components.feature_preprocessing
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *

# Custom wrapper class for using Sklearn's polynomial feature preprocessing
# function.
class custom_preprocessor(AutoSklearnPreprocessingAlgorithm):
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
        return {'shortname': 'CustomPreprocessor',
                'name': 'PolynomialFeatures',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # For each hyperparameter, its type (categorical, integer, float, etc.)
        # and its range and the default value must be specified here.
        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=2)
        interaction_only = CategoricalHyperparameter(
            name="interaction_only", choices=["False", "True"], default_value="False")
        include_bias = CategoricalHyperparameter(
            name="include_bias", choices=["True", "False"], default_value="True")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs


# Include the custom preprocessor class to auto-sklearn.
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(custom_preprocessor)

# Import toy data from sklearn and apply train_test_split.
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Run auto-sklearn regression with the custom preprocessor.
import autosklearn.regression
reg = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=30,
                                                  per_run_time_limit=10,
                                                  include_preprocessors=['custom_preprocessor']
                                                  )
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.show_models())
print(reg.sprint_statistics())

