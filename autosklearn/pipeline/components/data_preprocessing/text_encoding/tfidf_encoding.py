from typing import Dict, Optional, Tuple, Union

import itertools

import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class TfidfEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        ngram_range: str = "1,1",
        min_df: float = 0.0,
        max_df: float = 1.0,
        binary: bool = False,
        norm: str = "l2",
        sublinear_tf: bool = False,
        per_column: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.ngram_range = tuple(int(i) for i in ngram_range.split(","))
        self.random_state = random_state
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.per_column = per_column

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "TfidfEncoder":

        if isinstance(X, pd.DataFrame):
            X.fillna("", inplace=True)
            if self.per_column:
                self.preprocessor = {}

                for feature in X.columns:
                    vectorizer = TfidfVectorizer(
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        binary=self.binary,
                        norm=self.norm,
                        sublinear_tf=self.sublinear_tf,
                    ).fit(X[feature])
                    self.preprocessor[feature] = vectorizer
            else:
                self.preprocessor = TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_df=self.max_df,
                    binary=self.binary,
                    norm=self.norm,
                    sublinear_tf=self.sublinear_tf,
                )
                all_text = itertools.chain.from_iterable(X[col] for col in X.columns)
                self.preprocessor = self.preprocessor.fit(all_text)

        else:
            raise ValueError(
                "Your text data is not encoded in a pandas.DataFrame\n"
                "Please make sure to use a pandas.DataFrame and ensure"
                "that the text features are encoded as strings."
            )
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        X.fillna("", inplace=True)
        if self.per_column:
            X_new = None
            if self.preprocessor is None:
                raise NotImplementedError()

            for feature in self.preprocessor:
                # the names in the dataframe must not change
                if X_new is None:
                    X_new = self.preprocessor[feature].transform(X[feature])
                else:
                    X_transformed = self.preprocessor[feature].transform(X[feature])
                    X_new = hstack([X_new, X_transformed])

            return X_new
        else:
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
            "shortname": "RBOW",
            "name": "Relative Bag Of Word Encoder",
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
        hp_ngram_range = CSH.CategoricalHyperparameter(
            name="ngram_range",
            choices=["1,1", "1,2", "1,3", "2,2", "2,3", "3,3"],
            default_value="1,1",
        )

        hp_min_df = CSH.UniformFloatHyperparameter(
            # Todo this can still result in building no vectorizer
            name="min_df",
            lower=0.0,
            upper=0.3,
            default_value=0.0,
        )

        hp_max_df = CSH.UniformFloatHyperparameter(
            name="max_df", lower=0.7, upper=1.0, default_value=1.0
        )

        hp_binary = CSH.CategoricalHyperparameter(
            name="binary", choices=[True, False], default_value=False
        )

        hp_norm = CSH.CategoricalHyperparameter(
            name="norm", choices=["l2", "l1"], default_value="l2"
        )

        hp_sublinear_tf = CSH.CategoricalHyperparameter(
            name="sublinear_tf", choices=[True, False], default_value=False
        )

        hp_per_column = CSH.CategoricalHyperparameter(
            name="per_column", choices=[True, False], default_value=True
        )

        cs.add_hyperparameters(
            [
                hp_ngram_range,
                hp_max_df,
                hp_min_df,
                hp_binary,
                hp_norm,
                hp_sublinear_tf,
                hp_per_column,
            ]
        )

        return cs
