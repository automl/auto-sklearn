import numpy as np
import sklearn.ensemble
import sklearn.naive_bayes

from HPOlibConfigSpace.conditions import EqualsCondition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ..classification_base import ParamSklearnClassificationAlgorithm


class BaggedGaussianNB(ParamSklearnClassificationAlgorithm):

    def __init__(self, n_estimators, max_samples, max_features, random_state=None, verbose=0):

        self.n_estimators = n_estimators
        self.max_samples  = max_samples
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None

    def fit(self, X, Y):
        self.estimator = sklearn.ensemble.BaggingClassifier(base_estimator=sklearn.naive_bayes.GaussianNB(), n_estimators = self.n_estimators, max_samples = self.max_samples, max_features = self.max_features)
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
        return {'shortname': 'BaggedGaussianNB',
                'name': 'Bagging of Gaussian Naive Bayes classifiers',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):

        # The three parameters of the bagging ensemble are set to constants for now (SF)

        #UniformIntegerHyperparameter('n_estimators', lower=10, upper = 100)
        n_estimators = Constant('n_estimators', 100)

        #max_samples = UniformFloatHyperparameter('max_samples', lower = 0.5, upper=1.0)
        max_samples = Constant('max_samples' ,1.0)  # caution: has to be float!

        #max_features = UniformFloatHyperparameter('max_features', lower = 0.5, upper=1.0)
        max_features = Constant('max_features', 1.0) # caution: has to be float!

        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(max_samples)
        cs.add_hyperparameter(max_features)
        
        return cs

