from typing import Dict, Optional, Tuple, Union

import itertools

import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd
from ConfigSpace import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from sklearn.feature_extraction.text import CountVectorizer

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class BagOfWordEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        ngram_upper_bound: int = 1,
        min_df_choice: str = "min_df_absolute",
        min_df_absolute: int = 0,
        min_df_relative: float = 0.01,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.ngram_upper_bound = ngram_upper_bound
        self.random_state = random_state
        self.min_df_choice = min_df_choice
        self.min_df_absolute = min_df_absolute
        self.min_df_relative = min_df_relative

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "BagOfWordEncoder":

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "Your text data is not encoded in a pandas.DataFrame\n"
                "Please make sure to use a pandas.DataFrame and ensure"
                "that the text features are encoded as strings."
            )

        X.fillna("", inplace=True)

        # define a CountVectorizer for used on every feature
        if self.min_df_choice == "min_df_absolute":
            self.preprocessor = CountVectorizer(
                min_df=self.min_df_absolute,
                ngram_range=(1, self.ngram_upper_bound),
            )

        elif self.min_df_choice == "min_df_relative":
            self.preprocessor = CountVectorizer(
                min_df=self.min_df_relative,
                ngram_range=(1, self.ngram_upper_bound),
            )

        else:
            raise KeyError()

        all_text = itertools.chain.from_iterable(X[col] for col in X.columns)
        self.preprocessor = self.preprocessor.fit(all_text)

        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        X.fillna("", inplace=True)
        X_transformed = None
        if self.preprocessor is None:
            raise NotImplementedError()
        for feature in X.columns:
            if X_transformed is None:
                X_transformed = self.preprocessor.transform(X[feature])
            else:
                X_transformed += self.preprocessor.transform(X[feature])
        return X_transformed

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "BOW",
            "name": "Bag Of Word Encoder",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        hp_ngram_upper_bound = CSH.UniformIntegerHyperparameter(
            name="ngram_upper_bound", lower=1, upper=3, default_value=1
        )
        hp_min_df_choice_bow = CSH.CategoricalHyperparameter(
            "min_df_choice", choices=["min_df_absolute", "min_df_relative"]
        )
        hp_min_df_absolute_bow = CSH.UniformIntegerHyperparameter(
            name="min_df_absolute", lower=0, upper=10, default_value=0
        )
        hp_min_df_relative_bow = CSH.UniformFloatHyperparameter(
            name="min_df_relative", lower=0.01, upper=1.0, default_value=0.01, log=True
        )
        cs.add_hyperparameters(
            [
                hp_ngram_upper_bound,
                hp_min_df_choice_bow,
                hp_min_df_absolute_bow,
                hp_min_df_relative_bow,
            ]
        )

        cond_min_df_absolute_bow = EqualsCondition(
            hp_min_df_absolute_bow, hp_min_df_choice_bow, "min_df_absolute"
        )
        cond_min_df_relative_bow = EqualsCondition(
            hp_min_df_relative_bow, hp_min_df_choice_bow, "min_df_relative"
        )
        cs.add_conditions([cond_min_df_absolute_bow, cond_min_df_relative_bow])

        # maybe add bigrams ...

        return cs
