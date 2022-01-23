from typing import Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import EqualsCondition

import numpy as np
import pandas as pd
import itertools


from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT

from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        ngram_range: Optional[int] = None,
        min_df_choice: Optional[str] = None,
        min_df_absolute: Optional[int] = None,
        min_df_relative: Optional[float] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.min_df_choice = min_df_choice
        self.min_df_absolute = min_df_absolute
        self.min_df_relative = min_df_relative

    def fit(self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'BagOfWordEncoder':

        if isinstance(X, pd.DataFrame):
            # define a CountVectorizer for every feature (implicitly defined by order of columns,
            # maybe change the list
            # to a dictionary with features as keys)
            if self.min_df_choice == "min_df_absolute":
                self.preprocessor = CountVectorizer(min_df=self.min_df_absolute,
                                                    ngram_range=(1, self.ngram_range))
            elif self.min_df_choice == "min_df_relative":
                self.preprocessor = CountVectorizer(min_df=self.min_df_relative,
                                                    ngram_range=(1, self.ngram_range))
            else:
                raise KeyError()

            all_text = itertools.chain.from_iterable(X[col] for col in X.columns)
            self.preprocessor = self.preprocessor.fit(all_text)

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
        for feature in X.columns:
            if X_new is None:
                X_new = self.preprocessor.transform(X[feature])
            else:
                X_new += self.preprocessor.transform(X[feature])
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
                'handles_sparse': True,
                'handles_dense': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,), }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                                        ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        hp_ngram_range = CSH.UniformIntegerHyperparameter(name="ngram_range", lower=1, upper=3,
                                                          default_value=1)
        hp_min_df_choice_bow = CSH.CategoricalHyperparameter("min_df_choice",
                                                             choices=["min_df_absolute",
                                                                      "min_df_relative"])
        hp_min_df_absolute_bow = CSH.UniformIntegerHyperparameter(name="min_df_absolute", lower=0,
                                                                  upper=10,
                                                                  default_value=0)
        hp_min_df_relative_bow = CSH.UniformFloatHyperparameter(name="min_df_relative", lower=0.01,
                                                                upper=1.0,
                                                                default_value=0.01, log=True)
        cs.add_hyperparameters(
            [hp_ngram_range, hp_min_df_choice_bow, hp_min_df_absolute_bow, hp_min_df_relative_bow])

        cond_min_df_absolute_bow = EqualsCondition(hp_min_df_absolute_bow, hp_min_df_choice_bow,
                                                   "min_df_absolute")
        cond_min_df_relative_bow = EqualsCondition(hp_min_df_relative_bow, hp_min_df_choice_bow,
                                                   "min_df_relative")
        cs.add_conditions([cond_min_df_absolute_bow, cond_min_df_relative_bow])

        # maybe add bigrams ...

        return cs
