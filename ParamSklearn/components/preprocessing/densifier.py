from scipy import sparse

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ParamSklearn.components.preprocessor_base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.util import DENSE, SPARSE


class Densifier(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, densify, random_state):
        self.densify = bool(densify)
        self.random_state = random_state

    def fit(self, X, Y):
        return self

    def transform(self, X):
        return X.todense()

    @staticmethod
    def get_properties():
        return {'shortname': 'RandomTreesEmbedding',
                'name': 'Random Trees Embedding',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': False,
                'input': (SPARSE,),
                'output': DENSE,
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        densify = CategoricalHyperparameter("densify", ["False", "True"],
                                            default="False")
        cs.add_hyperparameter(densify)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %" % name

