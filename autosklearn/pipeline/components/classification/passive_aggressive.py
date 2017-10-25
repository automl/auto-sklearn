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
    def __init__(self, C, fit_intercept, tol, loss, random_state=None):
        self.C = float(C)
        self.fit_intercept = fit_intercept == 'True'
        self.tol = float(tol)
        self.loss = loss
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.iterative_fit(X, y, n_iter=1, refit=True)
        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1)

        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        from sklearn.linear_model.passive_aggressive import \
            PassiveAggressiveClassifier

        if refit:
            self.estimator = None

        if self.estimator is None:

            self.estimator = PassiveAggressiveClassifier(
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=1,
                loss=self.loss,
                shuffle=True,
                random_state=self.random_state,
                warm_start=True
            )
            self.classes_ = np.unique(y.astype(int))

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator.max_iter = 50
            self.estimator = sklearn.multiclass.OneVsRestClassifier(
                self.estimator, n_jobs=1)
            self.estimator.fit(X, y)
            self.fully_fit_ = True
        else:
            # In the first iteration, there is not yet an intercept

            self.estimator.max_iter += n_iter
            self.estimator.fit(X, y)
            if self.estimator.max_iter >= 50 or \
                            self.estimator.max_iter > self.estimator.n_iter_:
                self.fully_fit_ = True

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
        C = UniformFloatHyperparameter("C", 1e-5, 10, 1, log=True)
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        loss = CategoricalHyperparameter("loss",
                                         ["hinge", "squared_hinge"],
                                         default_value="hinge")

        tol = UniformFloatHyperparameter("tol", 1e-4, 1e-1, default_value=1e-3,
                                              log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([loss, fit_intercept, tol, C])
        return cs
