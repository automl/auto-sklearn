import warnings

import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA


class KernelPCA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, n_components, kernel, degree=3, gamma=0.25, coef0=0.0,
                 random_state=None):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, Y=None):
        import scipy.sparse
        import sklearn.decomposition

        self.n_components = int(self.n_components)
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        self.coef0 = float(self.coef0)

        self.preprocessor = sklearn.decomposition.KernelPCA(
            n_components=self.n_components, kernel=self.kernel,
            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            remove_zero_eig=True, random_state=self.random_state)
        if scipy.sparse.issparse(X):
            X = X.astype(np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            self.preprocessor.fit(X)
        # Raise an informative error message, equation is based ~line 249 in
        # kernel_pca.py in scikit-learn
        if len(self.preprocessor.alphas_ / self.preprocessor.lambdas_) == 0:
            raise ValueError('KernelPCA removed all features!')
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            X_new = self.preprocessor.transform(X)

            # TODO write a unittest for this case
            if X_new.shape[1] == 0:
                raise ValueError("KernelPCA removed all features!")

            return X_new

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'KernelPCA',
                'name': 'Kernel Principal Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': False,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100)
        kernel = CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
        gamma = UniformFloatHyperparameter(
            "gamma",
            3.0517578125e-05, 8,
            log=True,
            default_value=1.0,
        )
        degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, kernel, degree, gamma, coef0])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs
