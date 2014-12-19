from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ...implementations.StandardScaler import StandardScaler
from ...implementations.MinMaxScaler import MinMaxScaler
from ..preprocessor_base import AutoSklearnPreprocessingAlgorithm


class Rescaling(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, strategy, random_state=None):
        # TODO pay attention to the cases when a copy is made
        self.strategy = strategy

    def fit(self, X, Y):
        if self.strategy == "min/max":
            self.preprocessor = MinMaxScaler(copy=False)
        elif self.strategy == "standard":
            self.preprocessor = StandardScaler(copy=False)
        else:
            raise ValueError(self.strategy)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'Rescaling',
                'name': 'Rescaling',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                # Add something here...
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space():
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["min/max", "standard"], default="min/max")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "AutoSklearn %" % name
