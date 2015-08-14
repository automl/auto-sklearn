from sklearn.linear_model.ridge import RidgeClassifier

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition

from ParamSklearn.components.base import \
    ParamSklearnClassificationAlgorithm
from ParamSklearn.constants import *
from ParamSklearn.implementations.util import softmax


class Ridge(ParamSklearnClassificationAlgorithm):
    def __init__(self, alpha, fit_intercept, tol, class_weight=None,
                 random_state=None):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept == 'True'
        self.tol = float(tol)
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        self.estimator = RidgeClassifier(alpha=self.alpha,
                                         fit_intercept=self.fit_intercept,
                                         tol=self.tol,
                                         class_weight=self.class_weight,
                                         copy_X=False,
                                         normalize=False,
                                         solver='auto')
        self.estimator.fit(X, Y)
        return self

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
    def get_properties():
        return {'shortname': 'Rigde',
                'name': 'Rigde Classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': True,
                'input': (DENSE, SPARSE),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        alpha = cs.add_hyperparameter(UniformFloatHyperparameter(
            "alpha", 10 ** -5, 10., log=True, default=1.))
        fit_intercept = cs.add_hyperparameter(UnParametrizedHyperparameter(
            "fit_intercept", "True"))
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))
        return cs

    def __str__(self):
        return "ParamSklearn Ridge Classifier"
