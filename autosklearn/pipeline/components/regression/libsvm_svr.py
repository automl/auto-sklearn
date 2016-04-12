import resource

import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *


class LibSVM_SVR(AutoSklearnRegressionAlgorithm):
    def __init__(self, kernel, C, epsilon, tol, shrinking, gamma=0.0,
                 degree=3, coef0=0.0, verbose=False,
                 max_iter=-1, random_state=None):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.shrinking = shrinking
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft > 0:
                soft /= 1024 * 1024
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024
                cache_size = (soft - maxrss) / 1.5
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        self.C = float(self.C)
        self.epsilon = float(self.epsilon)
        self.tol = float(self.tol)
        self.shrinking = self.shrinking == 'True'
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.verbose = int(self.verbose)
        self.max_iter = int(self.max_iter)

        self.estimator = sklearn.svm.SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            tol=self.tol,
            shrinking=self.shrinking,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            cache_size=cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter
        )
        self.scaler = sklearn.preprocessing.StandardScaler(copy=True)

        self.scaler.fit(Y.reshape((-1, 1)))
        Y_scaled = self.scaler.transform(Y.reshape((-1, 1))).ravel()
        self.estimator.fit(X, Y_scaled)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        if self.scaler is None:
            raise NotImplementedError
        Y_pred = self.estimator.predict(X)
        return self.scaler.inverse_transform(Y_pred)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SVR',
                'name': 'Support Vector Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'prefers_data_normalized': True,
                'is_deterministic': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # Copied from libsvm_c
        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default=1.0)

        kernel = CategoricalHyperparameter(
            name="kernel", choices=['linear', 'poly', 'rbf', 'sigmoid'],
            default="rbf")
        degree = UniformIntegerHyperparameter(
            name="degree", lower=1, upper=5, default=3)

        # Changed the gamma value to 0.0 (is 0.1 for classification)
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=3.0517578125e-05, upper=8, log=True, default=0.1)

        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter(
            name="coef0", lower=-1, upper=1, default=0)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter(
            name="shrinking", choices=["True", "False"], default="True")
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default=1e-3, log=True)
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        # Random Guess
        epsilon = UniformFloatHyperparameter(name="epsilon", lower=0.001,
                                             upper=1, default=0.1, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(C)
        cs.add_hyperparameter(kernel)
        cs.add_hyperparameter(degree)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(coef0)
        cs.add_hyperparameter(shrinking)
        cs.add_hyperparameter(tol)
        cs.add_hyperparameter(max_iter)
        cs.add_hyperparameter(epsilon)

        degree_depends_on_kernel = InCondition(child=degree, parent=kernel,
                                               values=('poly', 'rbf', 'sigmoid'))
        gamma_depends_on_kernel = InCondition(child=gamma, parent=kernel,
                                              values=('poly', 'rbf'))
        coef0_depends_on_kernel = InCondition(child=coef0, parent=kernel,
                                              values=('poly', 'sigmoid'))
        cs.add_condition(degree_depends_on_kernel)
        cs.add_condition(gamma_depends_on_kernel)
        cs.add_condition(coef0_depends_on_kernel)
        return cs
