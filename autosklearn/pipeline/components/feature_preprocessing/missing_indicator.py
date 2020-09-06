import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, INPUT


class MissingIndicatorComponent(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, missing_values=np.nan, features: str = "missing-only",
                 random_state=None):
        super().__init__()
        self.features = features
        self.missing_values = missing_values
        self.random_state = random_state

    def fit(self, X, Y=None):
        from sklearn.impute import MissingIndicator
        self.preprocessor = MissingIndicator(missing_values=self.missing_values, features=self.features)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'MissingIndicator',
                'name': 'Missing Indicator',
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
        features = CategoricalHyperparameter("features", ["missing-only", "all"], default_value="missing-only")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([features])
        return cs
