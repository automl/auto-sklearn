from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT
from autosklearn.util.common import check_for_bool


class PolynomialFeatures(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, degree, interaction_only, include_bias, random_state=None):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.preprocessing

        self.degree = int(self.degree)
        self.interaction_only = check_for_bool(self.interaction_only)
        self.include_bias = check_for_bool(self.include_bias)

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
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # More than degree 3 is too expensive!
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only",
                                                     ["False", "True"], "False")
        include_bias = CategoricalHyperparameter("include_bias",
                                                 ["True", "False"], "True")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs
