import numpy as np
import sklearn.ensemble

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, Constant

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import SPARSE, DENSE, PREDICTIONS


class AdaboostClassifier(ParamSklearnClassificationAlgorithm):

    def __init__(self, n_estimators, learning_rate, algorithm='SAMME.R',
                 base_estimator=None, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)

        if algorithm not in ('SAMME.R', "SAMME"):
            raise ValueError("Illegal 'algorithm': %s" % algorithm)
        self.algorithm = algorithm
        self.random_state = random_state

        if base_estimator is None:
            self.base_estimator = base_estimator
        elif base_estimator == "None":
            self.base_estimator = None
        else:
            raise ValueError("Illegal ")

        self.estimator = None

    def fit(self, X, Y):
        self.estimator = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
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
        return {'shortname': 'AB',
                'name': 'AdaBoost Classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (SPARSE, DENSE),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                # But rather fortran or C-contiguous?
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.0001, upper=1, default=0.1, log=True)
        algorithm = Constant(name="algorithm", value="SAMME.R")
        base_estimator = Constant(name="base_estimator", value="None")

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=5, upper=50, default=10, log=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(learning_rate)
        cs.add_hyperparameter(base_estimator)
        cs.add_hyperparameter(algorithm)

        return cs

