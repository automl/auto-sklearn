import numpy as np
import sklearn.decomposition

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter

from ParamSklearn.components.preprocessor_base import ParamSklearnPreprocessingAlgorithm
from ParamSklearn.util import DENSE


class PCA(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, keep_variance, whiten, random_state=None):
        # TODO document that this implementation does not allow the number of
        #  components to be specified, but rather the amount of variance to
        # be kept!
        # TODO it would also be possible to use a heuristic for the number of
        #  PCA components!
        self.keep_variance = keep_variance
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X, Y):
        self.preprocessor = sklearn.decomposition.PCA(whiten=self.whiten,
                                                      copy=True)
        self.preprocessor.fit(X, Y)

        sum_ = 0.
        idx = 0
        while idx < len(self.preprocessor.explained_variance_ratio_) and \
                sum_ < self.keep_variance:
            sum_ += self.preprocessor.explained_variance_ratio_[idx]
            idx += 1

        components = self.preprocessor.components_
        self.preprocessor.components_ = components[:idx]

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
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
                'input': (DENSE, ),
                'output': DENSE,
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        keep_variance = UniformFloatHyperparameter(
            "keep_variance", 0.5, 1.0, default=1.0)
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "True"], default="False")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(keep_variance)
        cs.add_hyperparameter(whiten)
        return cs

    def __str__(self):
        name = self.get_properties()['name']
        return "ParamSklearn %s" % name
