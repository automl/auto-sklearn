import numpy as np
import sklearn.decomposition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from ParamSklearn.components.base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.constants import *


class PCA(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, keep_variance, whiten, random_state=None):
        self.keep_variance = keep_variance
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X, Y=None):
        n_components = float(self.keep_variance)
        self.preprocessor = sklearn.decomposition.PCA(n_components=n_components,
                                                      whiten=self.whiten,
                                                      copy=True)
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PCA',
                'name': 'Principle Component Analysis',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                # TODO write a test to make sure that the PCA scales data itself
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparsity...
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO document that we have to be very careful
                'is_deterministic': False,
                'handles_sparse': False,
                'handles_dense': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA),
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 0.9999, default=0.9999)
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default="False")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(keep_variance)
        cs.add_hyperparameter(whiten)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name
