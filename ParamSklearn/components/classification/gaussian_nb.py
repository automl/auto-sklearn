import numpy as np
import sklearn.naive_bayes

from HPOlibConfigSpace.configuration_space import ConfigurationSpace

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, PREDICTIONS


class GaussianNB(ParamSklearnClassificationAlgorithm):

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, Y):
        num_features = X.shape[1]
        self.estimator = sklearn.naive_bayes.GaussianNB()
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties():
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
                'input': (DENSE, ),
                'output': PREDICTIONS,
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs

