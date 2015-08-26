import numpy as np

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from ParamSklearn.components.base import ParamSklearnClassificationAlgorithm
from ParamSklearn.constants import *
from ParamSklearn.implementations import ProjLogit



class ProjLogitCLassifier(ParamSklearnClassificationAlgorithm):

    def __init__(self, max_epochs = 2, random_state=None, n_jobs=1):
        self.max_epochs = max_epochs
        self.estimator = None

    def fit(self, X, Y):
        self.estimator = ProjLogit.ProjLogit(max_epochs = int(self.max_epochs))
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
    def get_properties(dataset_properties=None):
        return {'shortname': 'PLogit',
                'name': 'Logistic Regresion using Least Squares',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': True,
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
        max_epochs = UniformIntegerHyperparameter("max_epochs", 1, 20, default=2)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(max_epochs)
        return cs
