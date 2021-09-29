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
        # ToDo define the Vecotizer

        sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt')
        l = sdatei.readlines()
        sdatei.close()
        l.append('\nX_fit:{}\n'.format(X, ))
        sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt', 'w')
        sdatei.write("".join(l))
        sdatei.close()

        if isinstance(X, pd.DataFrame):
            self.preprocessor = [CountVectorizer().fit(X[feature]) for feature in X.columns]
        else:
            raise NotImplementedError
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        X_new = None

        sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt')
        l = sdatei.readlines()
        sdatei.close()
        l.append('\nself.pre:{}\n'.format(self.preprocessor))
        sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt', 'w')
        sdatei.write("".join(l))
        sdatei.close()

        if self.preprocessor is None:
            raise NotImplementedError()
        for preprocessor, feature in zip(self.preprocessor, X.columns):
            if X_new is None:
                X_new = preprocessor.transform(X[feature])

                sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt')
                l = sdatei.readlines()
                sdatei.close()
                l.append(
                    '\nX_if:{}\n\nX_new: {}\nX_new_type: {}\nX_new_shape: {}\n\n'.format(X[feature], X_new, type(X_new),
                                                                                         X_new.shape))
                sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt', 'w')
                sdatei.write("".join(l))
                sdatei.close()

            else:
                X_new = hstack([X_new, preprocessor.transform(X[feature])])

                sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt')
                l = sdatei.readlines()
                sdatei.close()
                l.append('\nX_else:{}\n\nX_new: {}\nX_new_type: {}\nX_new_shape: {}\n\n'.format(X[feature], X_new,
                                                                                                type(X_new),
                                                                                                X_new.shape))
                sdatei = open('/home/lukas/Python_Projects/AutoSklearnDevelopment/sample.txt', 'w')
                sdatei.write("".join(l))
                sdatei.close()
        # return self.preprocessor.transform(X)
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
