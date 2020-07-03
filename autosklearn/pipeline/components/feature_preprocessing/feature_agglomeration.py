import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause

from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT


class FeatureAgglomeration(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, n_clusters, affinity, linkage, pooling_func,
                 random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.random_state = random_state

        self.pooling_func_mapping = dict(mean=np.mean,
                                         median=np.median,
                                         max=np.max)

    def fit(self, X, Y=None):
        import sklearn.cluster

        self.n_clusters = int(self.n_clusters)

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
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 400, 25)
        affinity = CategoricalHyperparameter(
            "affinity", ["euclidean", "manhattan", "cosine"], "euclidean")
        linkage = CategoricalHyperparameter(
            "linkage", ["ward", "complete", "average"], "ward")
        pooling_func = CategoricalHyperparameter(
            "pooling_func", ["mean", "median", "max"])

        cs.add_hyperparameters([n_clusters, affinity, linkage, pooling_func])

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs
