import numpy as np
import scipy.sparse

import autosklearn.pipeline.implementations.MinorityCoalescer

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool


class MinorityCoalescer(AutoSklearnPreprocessingAlgorithm):
    """ Group together categories which occurence is less than a specified mininum fraction.
    """

    def __init__(self, use_minimum_fraction=True, minimum_fraction=0.01):
        self.use_minimum_fraction = use_minimum_fraction
        self.minimum_fraction = minimum_fraction
        

    def _fit(self, X, y=None):
        self.use_minimum_fraction = check_for_bool(self.use_minimum_fraction)
        if self.use_minimum_fraction is False:
            self.minimum_fraction = None
        else:
            self.minimum_fraction = float(self.minimum_fraction)

        self.preprocessor = autosklearn.pipeline.implementations.MinorityCoalescer\
            .MinorityCoalescer(minimum_fraction=self.minimum_fraction)

        return self.preprocessor.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X)

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'coalescer',
                'name': 'Categorical minority coalescer',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        use_minimum_fraction = CategoricalHyperparameter(
            "use_minimum_fraction", ["True", "False"], default_value="True")
        minimum_fraction = UniformFloatHyperparameter(
            "minimum_fraction", lower=.0001, upper=0.5, default_value=0.01, log=True)
        cs.add_hyperparameters([use_minimum_fraction, minimum_fraction])
        cs.add_condition(EqualsCondition(minimum_fraction,
                                         use_minimum_fraction, 'True'))
        return cs
