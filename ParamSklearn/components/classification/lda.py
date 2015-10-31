import sklearn.lda
import sklearn.multiclass

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from HPOlibConfigSpace.conditions import EqualsCondition

from ParamSklearn.components.base import \
    ParamSklearnClassificationAlgorithm
from ParamSklearn.constants import *
from ParamSklearn.implementations.util import softmax


class LDA(ParamSklearnClassificationAlgorithm):
    def __init__(self, shrinkage, n_components, tol, shrinkage_factor=0.5,
        random_state=None):
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.tol = tol
        self.shrinkage_factor = shrinkage_factor
        self.estimator = None

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

        estimator = sklearn.lda.LDA(n_components=self.n_components,
                                    shrinkage=self.shrinkage,
                                    tol=self.tol,
                                    solver=solver)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
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
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        shrinkage = cs.add_hyperparameter(CategoricalHyperparameter(
            "shrinkage", ["None", "auto", "manual"], default="None"))
        shrinkage_factor = cs.add_hyperparameter(UniformFloatHyperparameter(
            "shrinkage_factor", 0., 1., 0.5))
        n_components = cs.add_hyperparameter(UniformIntegerHyperparameter(
            'n_components', 1, 250, default=10))
        tol = cs.add_hyperparameter(UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default=1e-4, log=True))

        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))
        return cs
