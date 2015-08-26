import sklearn.qda

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter

from ParamSklearn.components.base import \
    ParamSklearnClassificationAlgorithm
from ParamSklearn.constants import *
from ParamSklearn.implementations.util import softmax


class QDA(ParamSklearnClassificationAlgorithm):
    def __init__(self, reg_param, tol, random_state=None):
        self.reg_param = float(reg_param)
        self.tol = float(tol)
        self.estimator = None

    def fit(self, X, Y):

        self.estimator = sklearn.qda.QDA(self.reg_param)
        self.estimator.fit(X, Y, tol=self.tol)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        df = self.estimator.predict_proba(X)
        return softmax(df)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'QDA',
                'name': 'Quadratic Discriminant Analysis',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        reg_param = UniformFloatHyperparameter('reg_param', 0.0, 10.0,
                                                    default=0.5)
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default=1e-4,
                                         log=True)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(reg_param)
        cs.add_hyperparameter(tol)
        return cs
