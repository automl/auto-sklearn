import numpy as np
import pandas as pd

from autosklearn.pipeline.components.data_preprocessing.text_encoding.tfidf_encoding import (  # noqa: E501
    TfidfEncoder as Vectorizer,
)

import pytest


@pytest.mark.parametrize(
    "analyzer,per_column",
    [("word", True), ("word", False), ("char", True), ("char", False)],
)
def test_fit_transform(analyzer, per_column):
    X = pd.DataFrame(
        {
            "col1": ["hello world", "hello mars"],
            "col2": ["Test Test", "This is a test column"],
        }
    ).astype({"col1": "string", "col2": "string"})
    Vectorizer_fitted = Vectorizer(
        analyzer=analyzer,
        per_column=per_column,
        random_state=1,
    ).fit(X.copy())

    if per_column:
        for column in X.columns:
            Yt = Vectorizer_fitted.preprocessor[column].vocabulary_
            if column == "col1":
                if analyzer == "word":
                    words = sorted(
                        [
                            "hello",
                            "world",
                            "mars",
                        ]
                    )  # is ignored by TFIDFVectorizer
                    Y = {key: idx for idx, key in enumerate(words)}
                    assert Yt == Y
                else:
                    words = sorted(
                        [
                            "hell",
                            "ello",
                            "llo ",
                            "lo w",
                            "o wo",
                            " wor",
                            "worl",
                            "orld",
                            "lo m",
                            "o ma",
                            " mar",
                            "mars",
                        ]
                    )
                    Y = {key: idx for idx, key in enumerate(words)}
                    assert Yt == Y
            elif column == "col2":
                if analyzer == "word":
                    words = sorted(
                        [
                            "test",
                            "this",
                            "is",  # "a" is not added, len(...)=1,
                            "column",
                        ]
                    )  # is ignored by TFIDFVectorizer
                    Y = {key: idx for idx, key in enumerate(words)}
                    assert Yt == Y
                else:
                    words = sorted(
                        [
                            "test",
                            "est ",
                            "st t",
                            "t te",
                            " tes",
                            "this",
                            "his ",
                            "is i",
                            "s is",
                            " is ",
                            "is a",
                            "s a ",
                            " a t",
                            "a te",
                            "st c",
                            "t co",
                            " col",
                            "colu",
                            "olum",
                            "lumn",
                        ]
                    )
                    Y = {key: idx for idx, key in enumerate(words)}
                    assert Yt == Y
            else:
                raise ValueError(column)
    else:
        Yt = Vectorizer_fitted.preprocessor.vocabulary_
        if analyzer == "word":
            words = sorted(
                [
                    "hello",
                    "world",
                    "mars",
                    "test",
                    "this",
                    "is",  # "a" is not added, len(...)=1,
                    "column",
                ]
            )  # is ignored by TFIDFVectorizer
            Y = {key: idx for idx, key in enumerate(words)}
            assert Yt == Y
        else:
            words = sorted(
                [
                    "hell",
                    "ello",
                    "llo ",
                    "lo w",
                    "o wo",
                    " wor",
                    "worl",
                    "orld",
                    "lo m",
                    "o ma",
                    " mar",
                    "mars",
                    "test",
                    "est ",
                    "st t",
                    "t te",
                    " tes",
                    "this",
                    "his ",
                    "is i",
                    "s is",
                    " is ",
                    "is a",
                    "s a ",
                    " a t",
                    "a te",
                    "st c",
                    "t co",
                    " col",
                    "colu",
                    "olum",
                    "lumn",
                ]
            )
            Y = {key: idx for idx, key in enumerate(words)}
            assert Yt == Y


@pytest.mark.parametrize("per_column", [True, False])
def test_transform(per_column):
    X = pd.DataFrame(
        {
            "col1": ["hello world", "hello mars"],
            "col2": ["Test Test", "This is a test column"],
        }
    ).astype({"col1": "string", "col2": "string"})
    vectorizer = Vectorizer(
        per_column=per_column,
        analyzer="word",
        random_state=1,
    )
    X_t = vectorizer.fit_transform(X.copy())

    if per_column:
        # ['hello', 'mars', 'world', 'column', 'is', 'test', 'this']
        y = np.array(
            [
                [
                    0.57974,
                    0.0,
                    0.8148,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [0.57974, 0.8148, 0.0, 0.53405, 0.53405, 0.37998, 0.53405],
            ]
        )
        np.testing.assert_almost_equal(X_t.toarray(), y, decimal=5)
    else:
        print(vectorizer.preprocessor.vocabulary_)
        # 'column', 'hello', 'is', 'mars', 'test', 'this', 'world
        y = np.array(
            [
                [0.0, 0.61913, 0.0, 0.0, 1.0, 0.0, 0.78529],
                [0.52547, 0.61913, 0.52547, 0.78529, 0.41429, 0.52547, 0.0],
            ]
        )
        np.testing.assert_almost_equal(X_t.toarray(), y, decimal=5)


def test_check_shape():
    X = pd.DataFrame(
        {
            "col1": ["hello world", "this is test"],
            "col2": ["test test", "test test"],
        }
    ).astype({"col1": "string", "col2": "string"})
    X_t = Vectorizer(
        per_column=True,
        analyzer="word",
        random_state=1,
    ).fit_transform(X.copy())

    assert X_t.shape == (2, 6)

    X_t = Vectorizer(
        analyzer="word",
        per_column=False,
        random_state=1,
    ).fit_transform(X.copy())

    assert X_t.shape == (2, 5)


def test_check_nan():
    X = pd.DataFrame(
        {
            "col1": ["hello world", "this is test", None],
            "col2": ["test test", "test test", "test"],
        }
    ).astype({"col1": "string", "col2": "string"})
    X_t = Vectorizer(
        per_column=True,
        analyzer="word",
        random_state=1,
    ).fit_transform(X.copy())
    assert X_t.shape == (3, 6)

    X_t = Vectorizer(
        analyzer="word",
        per_column=False,
        random_state=1,
    ).fit_transform(X.copy())
    assert X_t.shape == (3, 5)


def test_check_vocabulary():
    X = pd.DataFrame(
        {
            "col1": ["hello world", "this is test", None],
            "col2": ["test test", "test test", "test"],
        }
    ).astype({"col1": "string", "col2": "string"})
    vectorizer = Vectorizer(
        per_column=True,
        analyzer="word",
        random_state=1,
    ).fit(X.copy())
    assert vectorizer.preprocessor["col1"].vocabulary_ == {
        "hello": 0,
        "world": 4,
        "this": 3,
        "is": 1,
        "test": 2,
    }
    assert vectorizer.preprocessor["col2"].vocabulary_ == {"test": 0}

    vectorizer = Vectorizer(
        analyzer="word",
        per_column=False,
        random_state=1,
    ).fit(X.copy())
    assert vectorizer.preprocessor.vocabulary_ == {
        "hello": 0,
        "world": 4,
        "this": 3,
        "is": 1,
        "test": 2,
    }
