import numpy as np
import sklearn.cluster

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from HPOlibConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause

from ParamSklearn.components.base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.constants import *


class FeatureAgglomeration(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, n_clusters, affinity, linkage, pooling_func,
        random_state=None):
        self.n_clusters = int(n_clusters)
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.random_state = random_state

        self.pooling_func_mapping = dict(mean=np.mean,
                                         median=np.median,
                                         max=np.max)

    def fit(self, X, Y=None):
        n_clusters = min(self.n_clusters, X.shape[1])
        if not callable(self.pooling_func):
            self.pooling_func = self.pooling_func_mapping[self.pooling_func]

        self.preprocessor = sklearn.cluster.FeatureAgglomeration(
            n_clusters=n_clusters, affinity=self.affinity,
            linkage=self.linkage, pooling_func=self.pooling_func)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Feature Agglomeration',
                'name': 'Feature Agglomeration',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                'prefers_data_normalized': True,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_clusters = cs.add_hyperparameter(UniformIntegerHyperparameter(
            "n_clusters", 2, 400, 25))
        affinity = cs.add_hyperparameter(CategoricalHyperparameter(
            "affinity", ["euclidean", "manhattan", "cosine"], "euclidean"))
        linkage = cs.add_hyperparameter(CategoricalHyperparameter(
            "linkage", ["ward", "complete", "average"], "ward"))
        pooling_func = cs.add_hyperparameter(CategoricalHyperparameter(
            "pooling_func", ["mean", "median", "max"]))

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs

