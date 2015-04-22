import sklearn.kernel_approximation

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from HPOlibConfigSpace.conditions import InCondition, EqualsCondition, AndConjunction

from ParamSklearn.components.preprocessor_base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.util import SPARSE, DENSE, INPUT


class Nystroem(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, kernel, n_components, gamma=None, degree=3,
                 coef0=1, random_state=None):
        self.kernel = kernel
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.random_state = random_state

    def fit(self, X, Y=None):
        self.preprocessor = sklearn.kernel_approximation.Nystroem(
            kernel=self.kernel, n_components=self.n_components,
            gamma=self.gamma, degree=self.degree, coef0=self.coef0,
            random_state=self.random_state)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'Nystroem',
                'name': 'Nystroem kernel approximation',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE),
                'output': INPUT,
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        kernel = CategoricalHyperparameter('kernel',
            ['chi2', 'poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default=0.1)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default=0)
        n_components = UniformIntegerHyperparameter(
            "n_components", 50, 10000, default=100, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(kernel)
        cs.add_hyperparameter(degree)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(coef0)
        cs.add_hyperparameter(n_components)

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf", "chi2",
                                                      "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
        cs.add_condition(gamma_condition)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name

