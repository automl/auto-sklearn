from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.algorithms import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.pipeline.implementations.util import softmax
import sklearn.discriminant_analysis
import sklearn.multiclass


class LDA(AutoSklearnClassificationAlgorithm):

    def __init__(self):
        super(LDA, self).__init__()
        self.shrinkage = None
        self.n_components = 10
        self.tol = 1e-4
        self.shrinkage_factor = 0.5

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        shrinkage = cs.add_hyperparameter(CategoricalHyperparameter(
            "shrinkage", ["None", "auto", "manual"], default="None"))
        shrinkage_factor = cs.add_hyperparameter(UniformFloatHyperparameter(
            "shrinkage_factor", 0., 1., default=0.5))
        n_components = cs.add_hyperparameter(UniformIntegerHyperparameter(
            'n_components', 1, 250, default=10))
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))

        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))

        return cs

    def fit(self, X, Y):

        if self.shrinkage == "None":
            self.shrinkage = None
            solver = 'svd'
        elif self.shrinkage == "auto":
            solver = 'lsqr'
        elif self.shrinkage == "manual":
            self.shrinkage = float(self.shrinkage_factor)
            solver = 'lsqr'
        else:
            raise ValueError(self.shrinkage)

        self.n_components = int(self.n_components)
        self.tol = float(self.tol)

        estimator = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=self.n_components, shrinkage=self.shrinkage,
            tol=self.tol, solver=solver)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict_proba(self, X):
        probas = super(LDA, self).predict_proba(X)
        probas = softmax(probas)
        return probas
