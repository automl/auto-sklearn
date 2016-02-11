from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class RandomTreesEmbedding(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, n_estimators, max_depth, min_samples_split,
                 min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes,
                 sparse_output=True, n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.ensemble

        if self.max_depth == "None":
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        if self.max_leaf_nodes == "None":
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.preprocessor = sklearn.ensemble.RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            sparse_output=self.sparse_output,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (SPARSE, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_estimators = UniformIntegerHyperparameter(name="n_estimators",
                                                    lower=10, upper=100,
                                                    default=10)
        max_depth = UniformIntegerHyperparameter(name="max_depth",
                                                 lower=2, upper=10,
                                                 default=5)
        min_samples_split = UniformIntegerHyperparameter(name="min_samples_split",
                                                         lower=2, upper=20,
                                                         default=2)
        min_samples_leaf = UniformIntegerHyperparameter(name="min_samples_leaf",
                                                        lower=1, upper=20,
                                                        default=1)
        min_weight_fraction_leaf = Constant('min_weight_fraction_leaf', 1.0)
        max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes",
                                                      value="None")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(min_weight_fraction_leaf)
        cs.add_hyperparameter(max_leaf_nodes)
        return cs
