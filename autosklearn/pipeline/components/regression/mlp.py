import copy
import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
    IterativeComponent,
)
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, PREDICTIONS
from autosklearn.util.common import check_for_bool


class MLPRegressor(
    IterativeComponent,
    AutoSklearnRegressionAlgorithm
):
    def __init__(self, hidden_layer_depth, num_nodes_per_layer, activation, alpha,
                 learning_rate_init, early_stopping, solver, batch_size,
                 n_iter_no_change, tol, shuffle, beta_1, beta_2, epsilon,
                 validation_fraction=None, random_state=None, verbose=0):
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.max_iter = self.get_max_iter()
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.solver = solver
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.beta_1 = beta_1

        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_iter_

    def iterative_fit(self, X, y, n_iter=2, refit=False):
        """
        Set n_iter=2 for the same reason as for SGD
        """
        from sklearn.neural_network import MLPRegressor
        import sklearn.preprocessing
        n_iter = max(n_iter, 2)

        if refit:
            self.estimator = None
            self.scaler = None

        if self.estimator is None:
            self._fully_fit = False

            self.max_iter = int(self.max_iter)
            self.hidden_layer_depth = int(self.hidden_layer_depth)
            self.num_nodes_per_layer = int(self.num_nodes_per_layer)
            self.hidden_layer_sizes = tuple(self.num_nodes_per_layer
                                            for i in range(self.hidden_layer_depth))
            self.activation = str(self.activation)
            self.alpha = float(self.alpha)
            self.learning_rate_init = float(self.learning_rate_init)
            self.early_stopping = str(self.early_stopping)
            if self.early_stopping == "train":
                self.validation_fraction = 0.0
                self.tol = float(self.tol)
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.early_stopping_val = False
            elif self.early_stopping == "valid":
                self.validation_fraction = float(self.validation_fraction)
                self.tol = float(self.tol)
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.early_stopping_val = True
            else:
                raise ValueError("Set early stopping to unknown value %s" % self.early_stopping)
            # elif self.early_stopping == "off":
            #     self.validation_fraction = 0
            #     self.tol = 10000
            #     self.n_iter_no_change = self.max_iter
            #     self.early_stopping_val = False

            self.solver = self.solver

            try:
                self.batch_size = int(self.batch_size)
            except ValueError:
                self.batch_size = str(self.batch_size)

            self.shuffle = check_for_bool(self.shuffle)
            self.beta_1 = float(self.beta_1)
            self.beta_2 = float(self.beta_2)
            self.epsilon = float(self.epsilon)
            self.beta_1 = float(self.beta_1)
            self.verbose = int(self.verbose)

            n_iter = int(np.ceil(n_iter))

            # initial fit of only increment trees
            self.estimator = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate_init=self.learning_rate_init,
                max_iter=n_iter,
                shuffle=self.shuffle,
                random_state=copy.copy(self.random_state),
                verbose=self.verbose,
                warm_start=True,
                early_stopping=self.early_stopping_val,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                beta_1=self.beta_2,
                beta_2=self.beta_1,
                epsilon=self.epsilon,
                # We do not use these, see comments below in search space
                # momentum=self.momentum,
                # nesterovs_momentum=self.nesterovs_momentum,
                # power_t=self.power_t,
                # learning_rate=self.learning_rate,
                # max_fun=self.max_fun
            )
            self.scaler = sklearn.preprocessing.StandardScaler(copy=True)
            self.scaler.fit(y.reshape((-1, 1)))
        else:
            new_max_iter = min(self.max_iter - self.estimator.n_iter_, n_iter)
            self.estimator.max_iter = new_max_iter

        Y_scaled = self.scaler.transform(y.reshape((-1, 1))).ravel()
        self.estimator.fit(X, Y_scaled)
        if self.estimator.n_iter_ >= self.max_iter or \
                self.estimator._no_improvement_count > self.n_iter_no_change:
            self._fully_fit = True
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, '_fully_fit'):
            return False
        else:
            return self._fully_fit

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        Y_pred = self.estimator.predict(X)
        return self.scaler.inverse_transform(Y_pred)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MLP',
                'name': 'Multilayer Percepton',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        hidden_layer_depth = UniformIntegerHyperparameter(name="hidden_layer_depth",
                                                          lower=1, upper=3, default_value=1)
        num_nodes_per_layer = UniformIntegerHyperparameter(name="num_nodes_per_layer",
                                                           lower=16, upper=264, default_value=32,
                                                           log=True)
        activation = CategoricalHyperparameter(name="activation", choices=['tanh', 'relu'],
                                               default_value='tanh')
        alpha = UniformFloatHyperparameter(name="alpha", lower=1e-7, upper=1e-1, default_value=1e-4,
                                           log=True)

        learning_rate_init = UniformFloatHyperparameter(name="learning_rate_init",
                                                        lower=1e-4, upper=0.5, default_value=1e-3,
                                                        log=True)

        # Not allowing to turn off early stopping
        early_stopping = CategoricalHyperparameter(name="early_stopping",
                                                   choices=["valid", "train"],  # , "off"],
                                                   default_value="valid")
        # Constants
        n_iter_no_change = Constant(name="n_iter_no_change", value=32)  # default=10 is too low
        validation_fraction = Constant(name="validation_fraction", value=0.1)
        tol = UnParametrizedHyperparameter(name="tol", value=1e-4)
        solver = Constant(name="solver", value='adam')

        # Relying on sklearn defaults for now
        batch_size = UnParametrizedHyperparameter(name="batch_size", value="auto")
        shuffle = UnParametrizedHyperparameter(name="shuffle", value="True")
        beta_1 = UnParametrizedHyperparameter(name="beta_1", value=0.9)
        beta_2 = UnParametrizedHyperparameter(name="beta_2", value=0.999)
        epsilon = UnParametrizedHyperparameter(name="epsilon", value=1e-8)

        # Not used
        # solver=["sgd", "lbfgs"] --> not used to keep searchspace simpler
        # learning_rate --> only used when using solver=sgd
        # power_t --> only used when using solver=sgd & learning_rate=invscaling
        # momentum --> only used when solver=sgd
        # nesterovs_momentum --> only used when solver=sgd
        # max_fun --> only used when solver=lbfgs
        # activation=["identity", "logistic"] --> not useful for classification

        cs.add_hyperparameters([hidden_layer_depth, num_nodes_per_layer,
                                activation, alpha,
                                learning_rate_init, early_stopping,
                                n_iter_no_change, validation_fraction, tol,
                                solver, batch_size, shuffle,
                                beta_1, beta_2, epsilon])

        validation_fraction_cond = InCondition(validation_fraction, early_stopping, ["valid"])
        cs.add_conditions([validation_fraction_cond])
        # We always use early stopping
        # n_iter_no_change_cond = InCondition(n_iter_no_change, early_stopping, ["valid", "train"])
        # tol_cond = InCondition(n_iter_no_change, early_stopping, ["valid", "train"])
        # cs.add_conditions([n_iter_no_change_cond, tol_cond])

        return cs
