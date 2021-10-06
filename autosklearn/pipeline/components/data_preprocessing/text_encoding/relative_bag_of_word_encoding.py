from typing import Dict, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import EqualsCondition

import scipy.sparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class RelativeBagOfWordEncoder(AutoSklearnPreprocessingAlgorithm):
    def __init__(
            self,
            use_idf=None,
            min_df_choice=None,
            min_df_absolute=None,
            min_df_relative=None,
            random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        self.random_state = random_state
        self.use_idf = use_idf
        self.min_df_choice = min_df_choice
        self.min_df_absolute = min_df_absolute
        self.min_df_relative = min_df_relative

    def fit(self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
            ) -> 'BagOfWordEncoder':

        if isinstance(X, pd.DataFrame):
            # define a CountVectorizer for every feature (implicitly defined by order of columns, maybe change the list
            # to a dictionary with features as keys)
            if self.min_df_choice == "min_df_absolute":
                self.preprocessor = CountVectorizer(min_df=self.min_df_absolute)
            elif self.min_df_choice == "min_df_relative":
                self.preprocessor = CountVectorizer(min_df=self.min_df_relative)
            else:
                raise KeyError()
            for feature in X.columns:
                self.preprocessor = self.preprocessor.fit(X[feature])
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
        X_new = TfidfTransformer(use_idf=self.use_idf).fit_transform(X_new)
        return X_new

    @staticmethod
    def get_properties(dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None
                       ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {'shortname': 'RBOW',
                'name': 'Relative Bag Of Word Encoder',
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

        hp_use_idf = CSH.CategoricalHyperparameter("use_idf", choices=[False, True])
        hp_min_df_choice = CSH.CategoricalHyperparameter("min_df_choice",
                                                         choices=["min_df_absolute", "min_df_relative"])
        hp_min_df_absolute = CSH.UniformIntegerHyperparameter(name="min_df_absolute", lower=0, upper=10,
                                                              default_value=0)
        hp_min_df_relative = CSH.UniformFloatHyperparameter(name="min_df_relative", lower=0.01, upper=1.0,
                                                            default_value=0.01, log=True)
        cs.add_hyperparameters([hp_use_idf, hp_min_df_choice, hp_min_df_absolute, hp_min_df_relative])

        cond_min_df_absolute = EqualsCondition(hp_min_df_absolute, hp_min_df_choice, "min_df_absolute")
        cond_min_df_relative = EqualsCondition(hp_min_df_relative, hp_min_df_choice, "min_df_relative")
        cs.add_conditions([cond_min_df_absolute, cond_min_df_relative])

        # maybe add bigrams ...

        return cs
