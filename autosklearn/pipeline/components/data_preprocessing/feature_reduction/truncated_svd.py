from typing import Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import EqualsCondition

import numpy as np
from scipy.sparse import csr_matrix

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
import autosklearn.pipeline.implementations.CategoryShift
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT

from sklearn.decomposition import TruncatedSVD


class FeatureReduction(AutoSklearnPreprocessingAlgorithm):
    """
    Reduces the features created by a bag of words encoding
    """

    def __init__(
        self,
        n_components=None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'FeatureReduction':
        X = csr_matrix(X)
        if X.shape[1] > int(2**self.n_components):
            self.preprocessor = TruncatedSVD(n_components=int(2**self.n_components))
            self.preprocessor.fit(X)
        elif X.shape[1] <= int(2**self.n_components) and X.shape[1] != 1:
            self.preprocessor = TruncatedSVD(n_components=int(2**(np.floor(np.log2(self.n_components)).astype(int))))
            self.preprocessor.fit(X)
        else:
            raise ValueError("The text embedding concistes only of a one dimension.\n"
                             "Are you sure that your text data is necessery ?\n")
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        if self.preprocessor is None:
            raise NotImplementedError()
        file = open("/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt", "a")
        file.write("\n X shape: {}\n\n".format(X.shape))
        file.close()

        file = open("/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt", "a")
        file.write("\n X_new shape: {}\n\n".format(self.preprocessor.transform(X).shape))
        file.close()

        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'FeatureReduction',
                'name': 'FeatureReduction',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter("n_components", lower=6, upper=10, default_value=7))
        return cs
