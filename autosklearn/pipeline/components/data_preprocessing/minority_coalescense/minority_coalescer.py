from typing import Dict, Optional, Tuple, Union


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np

import autosklearn.pipeline.implementations.MinorityCoalescer
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class MinorityCoalescer(AutoSklearnPreprocessingAlgorithm):
    """ Group together categories which occurence is less than a specified minimum fraction.
    """

    def __init__(self, minimum_fraction: float = 0.01,
                 random_state: Optional[np.random.RandomState] = None):
        self.minimum_fraction = minimum_fraction

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None
            ) -> 'MinorityCoalescer':
        self.minimum_fraction = float(self.minimum_fraction)

        self.preprocessor = autosklearn.pipeline.implementations.MinorityCoalescer\
            .MinorityCoalescer(minimum_fraction=self.minimum_fraction)
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'coalescer',
                'name': 'Categorical minority coalescer',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,), }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        minimum_fraction = UniformFloatHyperparameter(
            "minimum_fraction", lower=.0001, upper=0.5, default_value=0.01, log=True)
        cs.add_hyperparameter(minimum_fraction)
        return cs
