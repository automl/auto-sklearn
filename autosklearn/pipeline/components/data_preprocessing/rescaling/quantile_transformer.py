from typing import Dict, Optional, Tuple, Union

import numpy as np

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, SIGNED_DATA, SPARSE, INPUT
from autosklearn.pipeline.components.data_preprocessing.rescaling.abstract_rescaling \
    import Rescaling
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm


class QuantileTransformerComponent(Rescaling, AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        n_quantiles: int,
        output_distribution: str,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        from sklearn.preprocessing import QuantileTransformer
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.preprocessor = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            copy=False,
            random_state=random_state
        )

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'QuantileTransformer',
                'name': 'QuantileTransformer',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                # TODO find out if this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        # TODO parametrize like the Random Forest as n_quantiles = n_features^param
        n_quantiles = UniformIntegerHyperparameter(
            'n_quantiles', lower=10, upper=2000, default_value=1000
        )
        output_distribution = CategoricalHyperparameter(
            'output_distribution', ['uniform', 'normal']
        )
        cs.add_hyperparameters((n_quantiles, output_distribution))
        return cs
