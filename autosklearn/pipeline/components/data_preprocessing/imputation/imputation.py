from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.algorithms import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *


class Imputation(AutoSklearnPreprocessingAlgorithm):

    def __init__(self):
        # TODO pay attention to the cases when a copy is made (CSR matrices)
        self.strategy = 'mean'
        super(Imputation, self).__init__()

    def fit(self, X, y=None):
        import sklearn.preprocessing

        self.preprocessor = sklearn.preprocessing.Imputer(
            strategy=self.strategy, copy=False)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, SIGNED_DATA),
                'output': (INPUT,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["mean", "median", "most_frequent"], default="mean")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs
