import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.pipeline.implementations.util import softmax


class PassiveAggressive(AutoSklearnClassificationAlgorithm):
    def __init__(self, C, fit_intercept, n_iter, loss, random_state=None):
        self.C = float(C)
        self.fit_intercept = fit_intercept == 'True'
        self.n_iter = int(n_iter)
        self.loss = loss
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1)

        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        from sklearn.linear_model.passive_aggressive import \
            PassiveAggressiveClassifier

        if refit:
            self.estimator = None

        if self.estimator is None:
            self._iterations = 0

            self.estimator = PassiveAggressiveClassifier(
                C=self.C, fit_intercept=self.fit_intercept, n_iter=1,
                loss=self.loss, shuffle=True, random_state=self.random_state,
                warm_start=True)
            self.classes_ = np.unique(y.astype(int))

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator.n_iter = self.n_iter
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
            self.estimator.fit(X, y)
            self.fully_fit_ = True
        else:
            # In the first iteration, there is not yet an intercept

            self.estimator.n_iter = n_iter
            self.estimator.partial_fit(X, y, classes=np.unique(y))
            if self._iterations >= self.n_iter:
                self.fully_fit_ = True
            self._iterations += n_iter

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.decision_function(X)
        return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PassiveAggressive Classifier',
                'name': 'Passive Aggressive Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        loss = CategoricalHyperparameter("loss",
                                         ["hinge", "squared_hinge"],
                                         default="hinge")
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        n_iter = UniformIntegerHyperparameter("n_iter", 5, 1000, default=20,
                                              log=True)
        C = UniformFloatHyperparameter("C", 1e-5, 10, 1, log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(loss)
        cs.add_hyperparameter(fit_intercept)
        cs.add_hyperparameter(n_iter)
        cs.add_hyperparameter(C)
        return cs
