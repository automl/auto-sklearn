import numpy as np
import sklearn.svm

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.conditions import EqualsCondition, OrConjunction, \
    InCondition
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter


from ParamSklearn.components.regression_base import ParamSklearnRegressionAlgorithm
from ParamSklearn.util import DENSE, SPARSE, PREDICTIONS

# Something is wrong here...
"""
class SupportVectorRegression(ParamSklearnRegressionAlgorithm):
    def __init__(self, kernel, C, epsilon, degree, coef0, tol, shrinking,
                 gamma=0.0, probability=False, cache_size=2000, verbose=False,
                 max_iter=-1, random_state=None
                 ):

        if kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
            self.kernel = kernel
        else:
            raise ValueError("'kernel' must be in ('linear', 'poly', 'rbf', "
                             "'sigmoid'), but is %s" % str(kernel))
        self.gamma = float(gamma)
        self.C = float(C)
        self.epsilon = epsilon
        self.degree = int(float(degree))
        self.coef0 = float(coef0)
        self.tol = float(tol)

        if shrinking == "True":
            self.shrinking = True
        elif shrinking == "False":
            self.shrinking = False
        else:
            raise ValueError("'shrinking' must be in ('True', 'False'), "
                             "but is %s" % str(shrinking))

        # We don't assume any hyperparameters here
        self.probability = probability
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = int(float(max_iter))
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):

        self.estimator = sklearn.svm.SVR(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            probability=self.probability,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'SVR',
                'name': 'Support Vector Regression',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # TODO find out if this is good because of sparcity...
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'prefers_data_normalized': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (SPARSE, DENSE),
                'ouput': PREDICTIONS,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties):
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
"""
