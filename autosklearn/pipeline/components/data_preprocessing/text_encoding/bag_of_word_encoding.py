from typing import Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace

import scipy.sparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT

from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(
            self,
            random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.random_state = random_state

    def fit(self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'BagOfWordEncoder':

        if isinstance(X, pd.DataFrame):
            # define a CountVectorizer for every feature (implicitly defined by order of columns, maybe change the list
            # to a dictionary with features as keys)
            self.preprocessor = [CountVectorizer().fit(X[feature]) for feature in X.columns]
        else:
            raise ValueError("Your text data is not encoded in a pandas.DataFrame\n"
                             "Please make sure to use a pandas.DataFrame and ensure"
                             "that the text features are encoded as strings.")
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        X_new = None
        if self.preprocessor is None:
            raise NotImplementedError()
        # iterate over the pretrained preprocessors and columns and transform the data
        for preprocessor, feature in zip(self.preprocessor, X.columns):
            if X_new is None:
                X_new = preprocessor.transform(X[feature])
            else:
                X_new = hstack([X_new, preprocessor.transform(X[feature])])
        return X_new

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'BOW',
                'name': 'Bag Of Word Encoder',
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
        return ConfigurationSpace()
