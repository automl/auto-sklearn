import numpy as np
import scipy.sparse

import autosklearn.pipeline.implementations.OneHotEncoder

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_for_bool, check_none


class OneHotEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, use_minimum_fraction=True, minimum_fraction=0.01,
                 categorical_features=None, random_state=None):
        # TODO pay attention to the cases when a copy is made (CSR matrices)

        self.use_minimum_fraction = use_minimum_fraction
        self.minimum_fraction = minimum_fraction
        self.categorical_features = categorical_features

    def _fit(self, X, y=None):
        self.use_minimum_fraction = check_for_bool(self.use_minimum_fraction)
        if self.use_minimum_fraction is False:
            self.minimum_fraction = None
        else:
            self.minimum_fraction = float(self.minimum_fraction)

        if check_none(self.categorical_features):
            categorical_features = []
        else:
            categorical_features = self.categorical_features

        self.preprocessor = autosklearn.pipeline.implementations.OneHotEncoder\
            .OneHotEncoder(minimum_fraction=self.minimum_fraction,
                           categorical_features=categorical_features,
                           sparse=True)

        return self.preprocessor.fit_transform(X)

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        is_sparse = scipy.sparse.issparse(X)
        X = self._fit(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

    def transform(self, X):
        is_sparse = scipy.sparse.issparse(X)
        if self.preprocessor is None:
            raise NotImplementedError()
        X = self.preprocessor.transform(X)
        if is_sparse:
            return X
        elif isinstance(X, np.ndarray):
            return X
        else:
            return X.toarray()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
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
