import sklearn.decomposition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration

from HPOlibConfigSpace.hyperparameters import  IntegerHyperparameter

from ..preprocessor_base import AutoSklearnPreprocessingAlgorithm
import numpy as np



class TruncatedSVD(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, target_dim, random_state=None):
        # TODO: fill out handles_???
        #		how to set the maximum of the hyperparameter search space for target dim in a meaningful way?
        self.target_dim = target_dim
        self.random_state = random_state
        self.preprocessor=None

    def fit(self, X, Y):
        self.preprocessor = sklearn.decomposition.TruncatedSVD(min(self.target_dim, X.shape[0]), algorithm='arpack')
        self.preprocessor.fit(X, Y)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'TSVD',
                'name': 'Truncated Singular Value Decomposition',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': None,
                'handles_classification': None,
                'handles_multiclass': None,
                'handles_multilabel': None,
                'is_deterministic': True,
                'handles_sparse': True,
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        target_dim = IntegerHyperparameter(
            "target_dim", 0, 256, default=128)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(target_dim)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "AutoSklearn %" % name
