import numpy as np
import sklearn.naive_bayes

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, SPARSE, PREDICTIONS


class MultinomialNB(ParamSklearnClassificationAlgorithm):

    def __init__(self, alpha, fit_prior, random_state=None, verbose=0):
        self.alpha = alpha
        if fit_prior.lower() == "true":
            self.fit_prior = True
        elif fit_prior.lower() == "false":
            self.fit_prior = False
        else:
            self.fit_prior = fit_prior

        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, Y):
        self.estimator = sklearn.naive_bayes.MultinomialNB(alpha=self.alpha,
            fit_prior=self.fit_prior)
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
        return {'shortname': 'MultinomialNB',
                'name': 'Multinomial Naive Bayes classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                # sklearn website says:  The multinomial distribution normally
                # requires integer feature counts. However, in practice,
                # fractional counts such as tf-idf may also work.
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, SPARSE),
                'output': PREDICTIONS,
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        
        # the smoothing parameter is a non-negative float
        # I will limit it to 100 and put it on a logarithmic scale. (SF)
        # Please adjust that, if you know a proper range, this is just a guess.
        alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
                                           default=1, log=True)

        fit_prior = CategoricalHyperparameter( name="fit_prior",
                                               choices=["True", "False"],
                                               default="True")
        
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(fit_prior)
        
        return cs

