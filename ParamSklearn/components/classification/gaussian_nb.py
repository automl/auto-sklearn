import numpy as np
import sklearn.naive_bayes

from HPOlibConfigSpace.configuration_space import ConfigurationSpace

from ParamSklearn.components.base import ParamSklearnClassificationAlgorithm
from ParamSklearn.constants import *


class GaussianNB(ParamSklearnClassificationAlgorithm):

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, y):
        while not self.configuration_fully_fitted():
            self.iterative_fit(X, y, n_iter=1)
        return self

    def iterative_fit(self, X, y, n_iter=1, refit=False):
        if refit:
            self.estimator = None

        if self.estimator is None:
            self.n_iter = 0
            self.fully_fit_ = False
            self.estimator = sklearn.naive_bayes.GaussianNB()
            self.classes_ = np.unique(y.astype(int))

        for iter in range(n_iter):
            start = min(self.n_iter * 1000, y.shape[0])
            stop = min((self.n_iter + 1) * 1000, y.shape[0])
            self.estimator.partial_fit(X[start:stop], y[start:stop],
                                       self.classes_)
            self.n_iter += 1

            if stop >= len(y):
                self.fully_fit_ = True
                break

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
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GaussianNB',
                'name': 'Gaussian Naive Bayes classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

