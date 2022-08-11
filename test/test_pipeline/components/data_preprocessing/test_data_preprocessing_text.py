import numpy as np
import pandas as pd

from autosklearn.pipeline.components.data_preprocessing.text_encoding.tfidf_encoding import (  # noqa: E501
    TfidfEncoder as Vectorizer,
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
        Vectorizer_fitted = Vectorizer(
            per_column=False,
            random_state=1,
        ).fit(X.copy())

        Yt = Vectorizer_fitted.preprocessor.vocabulary_
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

        Vectorizer_fitted = Vectorizer(
            per_column=True,
            random_state=1,
        ).fit(X.copy())

        for key in Vectorizer_fitted.preprocessor:
            y = []
            for col in X[key]:
                y += [word for word in col.lower().split(" ") if len(word) > 1]
            y = sorted(y)
            yt = sorted(Vectorizer_fitted.preprocessor[key].vocabulary_.keys())
            np.testing.assert_array_equal(yt, y)

    def test_transform(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is a test"],
                "col2": ["hello mars", "this is the second column"],
            }
        ).astype({"col1": "string", "col2": "string"})
        X_t = Vectorizer(
            random_state=1,
        ).fit_transform(X.copy())

        # ['column', 'hello', 'is', 'mars', 'second', 'test', 'the', 'this', 'world']
        y = np.array(
            [
                [
                    0.707107,
                    0.0,
                    0.0,
                    0.0,
                    0.707107,
                    0.0,
                    0.707107,
                    0.0,
                    0.707107,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.57735,
                    0.57735,
                    0.57735,
                    0.0,
                    0.447214,
                    0.0,
                    0.447214,
                    0.0,
                    0.447214,
                    0.447214,
                    0.447214,
                ],
            ]
        )
        np.testing.assert_almost_equal(X_t.toarray(), y, decimal=5)

        X_t = Vectorizer(
            per_column=False,
            random_state=1,
        ).fit_transform(X.copy())

        # 'hello', 'is', 'test', 'this', 'world',
        # 'column', 'hello', 'is', 'mars', 'second', 'the', 'this'
        y = np.array(
            [
                [0.0, 1.238261, 0.0, 0.785288, 0.0, 0.0, 0.0, 0.0, 0.785288],
                [
                    0.485461,
                    0.0,
                    0.909148,
                    0.0,
                    0.485461,
                    0.667679,
                    0.485461,
                    0.909148,
                    0.0,
                ],
            ]
        )
        np.testing.assert_almost_equal(X_t.toarray(), y, decimal=5)

    def test_check_shape(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is test"],
                "col2": ["test test", "test test"],
            }
        ).astype({"col1": "string", "col2": "string"})
        X_t = Vectorizer(
            random_state=1,
        ).fit_transform(X.copy())

        self.assertEqual(X_t.shape, (2, 6))

        X_t = Vectorizer(
            per_column=False,
            random_state=1,
        ).fit_transform(X.copy())

        self.assertEqual(X_t.shape, (2, 5))

    def test_check_nan(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is test", None],
                "col2": ["test test", "test test", "test"],
            }
        ).astype({"col1": "string", "col2": "string"})
        X_t = Vectorizer(
            random_state=1,
        ).fit_transform(X.copy())
        self.assertEqual(X_t.shape, (3, 6))

        X_t = Vectorizer(
            per_column=False,
            random_state=1,
        ).fit_transform(X.copy())
        self.assertEqual(X_t.shape, (3, 5))

    def test_check_vocabulary(self):
        X = pd.DataFrame(
            {
                "col1": ["hello world", "this is test", None],
                "col2": ["test test", "test test", "test"],
            }
        ).astype({"col1": "string", "col2": "string"})
        vectorizer = Vectorizer(
            random_state=1,
        ).fit(X.copy())
        self.assertEqual(
            vectorizer.preprocessor["col1"].vocabulary_,
            {"hello": 0, "world": 4, "this": 3, "is": 1, "test": 2},
        )
        self.assertEqual(vectorizer.preprocessor["col2"].vocabulary_, {"test": 0})

        vectorizer = Vectorizer(
            per_column=False,
            random_state=1,
        ).fit(X.copy())
        self.assertEqual(
            vectorizer.preprocessor.vocabulary_,
            {"hello": 0, "world": 4, "this": 3, "is": 1, "test": 2},
        )
