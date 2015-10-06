import sklearn.preprocessing

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from ParamSklearn.components.base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.constants import *


class PolynomialFeatures(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, degree, interaction_only, include_bias, random_state=None):
        self.degree = int(degree)
        self.interaction_only = interaction_only.lower() == 'true'
        self.include_bias = include_bias.lower() == 'true'
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only,
            include_bias=self.include_bias)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PolynomialFeatures',
                'name': 'PolynomialFeatures',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                # this here suggests so http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
                'handles_sparse': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # More than degree 3 is too expensive!
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only",
                                                     ["False", "True"], "False")
        include_bias = CategoricalHyperparameter("include_bias",
                                                 ["True", "False"], "True")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(degree)
        cs.add_hyperparameter(interaction_only)
        cs.add_hyperparameter(include_bias)

        return cs
