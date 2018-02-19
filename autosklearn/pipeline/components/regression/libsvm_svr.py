import resource
import sys

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none

class LibSVM_SVR(AutoSklearnRegressionAlgorithm):
    def __init__(self, kernel, C, epsilon, tol, shrinking, gamma=0.1,
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

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM. The cache size is
        # calculated as 2/3 of the available memory (which is calculated as the memory limit minus
        # the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

                # In MacOS, the MaxRSS output of resource.getrusage in bytes; on other platforms,
                # it's in kilobytes
                if sys.platform == 'darwin':
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        self.C = float(self.C)
        self.epsilon = float(self.epsilon)
        self.tol = float(self.tol)
        self.shrinking = check_for_bool(self.shrinking)
        self.degree = int(self.degree)
        self.gamma = float(self.gamma)
        if check_none(self.coef0):
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
        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default_value=1.0)
        # Random Guess
        epsilon = UniformFloatHyperparameter(name="epsilon", lower=0.001,
                                             upper=1, default_value=0.1,
                                             log=True)

        kernel = CategoricalHyperparameter(
            name="kernel", choices=['linear', 'poly', 'rbf', 'sigmoid'],
            default_value="rbf")
        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=3)

        gamma = UniformFloatHyperparameter(
            name="gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)

        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter(
            name="coef0", lower=-1, upper=1, default_value=0)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter(
            name="shrinking", choices=["True", "False"], default_value="True")
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                               tol, max_iter, epsilon])

        degree_depends_on_kernel = InCondition(child=degree, parent=kernel,
                                               values=('poly', 'rbf', 'sigmoid'))
        gamma_depends_on_kernel = InCondition(child=gamma, parent=kernel,
                                              values=('poly', 'rbf'))
        coef0_depends_on_kernel = InCondition(child=coef0, parent=kernel,
                                              values=('poly', 'sigmoid'))
        cs.add_conditions([degree_depends_on_kernel, gamma_depends_on_kernel,
                           coef0_depends_on_kernel])

        return cs
