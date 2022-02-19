from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UnParametrizedHyperparameter,
)

from autosklearn.pipeline.components.base import (
    AutoSklearnRegressionAlgorithm,
    IterativeComponent,
)
from autosklearn.pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from autosklearn.util.common import check_for_bool


class SGD(
    IterativeComponent,
    AutoSklearnRegressionAlgorithm,
):
    def __init__(
        self,
        loss,
        penalty,
        alpha,
        fit_intercept,
        tol,
        learning_rate,
        l1_ratio=0.15,
        epsilon=0.1,
        eta0=0.01,
        power_t=0.5,
        average=False,
        random_state=None,
    ):
        self.max_iter = self.get_max_iter()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon = epsilon
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average

        self.estimator = None
        self.scaler = None

    @staticmethod
    def get_max_iter():
        return 1024

    def get_current_iter(self):
        return self.n_iter_

    def iterative_fit(self, X, y, n_iter=2, refit=False):
        import sklearn.preprocessing
        from sklearn.linear_model import SGDRegressor

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.
        n_iter = max(n_iter, 2)

        if refit:
            self.estimator = None
            self.scaler = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.alpha = float(self.alpha)
            self.l1_ratio = float(self.l1_ratio) if self.l1_ratio is not None else 0.15
            self.epsilon = float(self.epsilon) if self.epsilon is not None else 0.1
            self.eta0 = float(self.eta0)
            self.power_t = float(self.power_t) if self.power_t is not None else 0.25
            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.tol = float(self.tol)

            self.estimator = SGDRegressor(
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=n_iter,
                tol=self.tol,
                learning_rate=self.learning_rate,
                l1_ratio=self.l1_ratio,
                epsilon=self.epsilon,
                eta0=self.eta0,
                power_t=self.power_t,
                shuffle=True,
                average=self.average,
                random_state=self.random_state,
                warm_start=True,
            )

            self.scaler = sklearn.preprocessing.StandardScaler(copy=True)

            if y.ndim == 1:
                y = y.reshape((-1, 1))

            y_scaled = self.scaler.fit_transform(y)

            # Flatten: [[0], [0], [0]] -> [0, 0, 0]
            if y_scaled.ndim == 2 and y_scaled.shape[1] == 1:
                y_scaled = y_scaled.flatten()

            self.estimator.fit(X, y_scaled)
            self.n_iter_ = self.estimator.n_iter_
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, self.max_iter)

            # Convert y to be at least 2d for the scaler
            # [1,1,1] -> [[1], [1], [1]]
            if y.ndim == 1:
                y = y.reshape((-1, 1))

            y_scaled = self.scaler.transform(y)

            # Flatten: [[0], [0], [0]] -> [0, 0, 0]
            if y_scaled.ndim == 2 and y_scaled.shape[1] == 1:
                y_scaled = y_scaled.flatten()

            self.estimator._validate_params()
            self.estimator._partial_fit(
                X,
                y_scaled,
                alpha=self.estimator.alpha,
                C=1.0,
                loss=self.estimator.loss,
                learning_rate=self.estimator.learning_rate,
                max_iter=n_iter,
                sample_weight=None,
                coef_init=None,
                intercept_init=None,
            )
            self.n_iter_ += self.estimator.n_iter_

        if (
            self.estimator.max_iter >= self.max_iter
            or self.estimator.max_iter > self.n_iter_
        ):
            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, "fully_fit_"):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        Y_pred = self.estimator.predict(X)
        return self.scaler.inverse_transform(Y_pred)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "SGD Regressor",
            "name": "Stochastic Gradient Descent Regressor",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "handles_sparse": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        loss = CategoricalHyperparameter(
            "loss",
            [
                "squared_loss",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ],
            default_value="squared_loss",
        )
        penalty = CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default_value="l2"
        )
        alpha = UniformFloatHyperparameter(
            "alpha", 1e-7, 1e-1, log=True, default_value=0.0001
        )
        l1_ratio = UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1.0, log=True, default_value=0.15
        )
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True
        )
        epsilon = UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default_value=0.1, log=True
        )
        learning_rate = CategoricalHyperparameter(
            "learning_rate",
            ["optimal", "invscaling", "constant"],
            default_value="invscaling",
        )
        eta0 = UniformFloatHyperparameter(
            "eta0", 1e-7, 1e-1, default_value=0.01, log=True
        )
        power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, default_value=0.25)
        average = CategoricalHyperparameter(
            "average", ["False", "True"], default_value="False"
        )

        cs.add_hyperparameters(
            [
                loss,
                penalty,
                alpha,
                l1_ratio,
                fit_intercept,
                tol,
                epsilon,
                learning_rate,
                eta0,
                power_t,
                average,
            ]
        )

        # TODO add passive/aggressive here, although not properly documented?
        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = InCondition(
            epsilon,
            loss,
            ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        )

        # eta0 is only relevant if learning_rate!='optimal' according to code
        # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
        # linear_model/sgd_fast.pyx#L603
        eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling", "constant"])
        power_t_condition = EqualsCondition(power_t, learning_rate, "invscaling")

        cs.add_conditions(
            [elasticnet, epsilon_condition, power_t_condition, eta0_in_inv_con]
        )

        return cs
