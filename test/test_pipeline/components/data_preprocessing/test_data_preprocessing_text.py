import numpy as np
import pandas as pd

from autosklearn.pipeline.components.data_preprocessing.text_encoding.bag_of_word_encoding import (  # noqa: E501
    BagOfWordEncoder as BOW,
)
from autosklearn.pipeline.components.data_preprocessing.text_encoding.bag_of_word_encoding_distinct import (  # noqa: E501
    BagOfWordEncoder as BOW_distinct,
)

import unittest


class TextPreprocessingPipelineTest(unittest.TestCase):
    def test_fit_transform(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "This is a test"],
                "col2": ["hello mars", "This is the second column"],
            }
        ).astype({"col1": "string", "col2": "string"})
        BOW_fitted = BOW(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit(X.copy())

        Yt = BOW_fitted.preprocessor.vocabulary_
        words = sorted(
            [
                "hello",
                "world",
                "this",
                "is",
                "test",  # "a" is not added, len(...)=1
                "mars",
                "the",
                "second",
                "column",
            ]
        )  # is ignored by CountVectorizer
        Y = {key: idx for idx, key in enumerate(words)}

        np.testing.assert_array_equal(Yt, Y)

        BOW_fitted = BOW_distinct(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit(X.copy())

        for key in BOW_fitted.preprocessor:
            y = []
            for col in X[key]:
                y += [word for word in col.lower().split(" ") if len(word) > 1]
            y = sorted(y)
            yt = sorted(BOW_fitted.preprocessor[key].vocabulary_.keys())
            np.testing.assert_array_equal(yt, y)

    def test_transform(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is a test"],
                "col2": ["hello mars", "this is the second column"],
            }
        ).astype({"col1": "string", "col2": "string"})
        X_t = BOW(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit_transform(X.copy())

        # ['column', 'hello', 'is', 'mars', 'second', 'test', 'the', 'this', 'world']
        y = np.array([[0, 2, 0, 1, 0, 0, 0, 0, 1], [1, 0, 2, 0, 1, 1, 1, 2, 0]])
        np.testing.assert_array_equal(X_t.toarray(), y)

        X_t = BOW_distinct(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit_transform(X.copy())

        # 'hello', 'is', 'test', 'this', 'world',
        # 'column', 'hello', 'is', 'mars', 'second', 'the', 'this'
        y = np.array(
            [[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]]
        )
        np.testing.assert_array_equal(X_t.toarray(), y)

    def test_check_shape(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is test"],
                "col2": ["test test", "test test"],
            }
        ).astype({"col1": "string", "col2": "string"})
        X_t = BOW(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit_transform(X.copy())

        self.assertEqual(X_t.shape, (2, 5))

        X_t = BOW_distinct(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit_transform(X.copy())

        self.assertEqual(X_t.shape, (2, 6))

    def test_check_nan(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is test", None],
                "col2": ["test test", "test test", "test"],
            }
        ).astype({"col1": "string", "col2": "string"})
        X_t = BOW(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit_transform(X.copy())

        self.assertEqual(X_t.shape, (3, 5))

        X_t = BOW_distinct(
            ngram_upper_bound=1,
            min_df_choice="min_df_absolute",
            min_df_absolute=0,
            min_df_relative=0,
            random_state=1,
        ).fit_transform(X.copy())

        self.assertEqual(X_t.shape, (3, 6))
