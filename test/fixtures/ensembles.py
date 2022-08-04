from __future__ import annotations

from typing import Callable, Collection, Optional, Union

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor

from autosklearn.data.validation import (
    SUPPORTED_FEAT_TYPES,
    SUPPORTED_TARGET_TYPES,
    InputValidator,
)
from autosklearn.evaluation.abstract_evaluator import (
    MyDummyClassifier,
    MyDummyRegressor,
)
from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    AutoSklearnRegressionAlgorithm,
)

from pytest_cases import fixture

from test.conftest import DEFAULT_SEED


@fixture
def make_voting_classifier() -> Callable[..., VotingClassifier]:
    """
    Parameters
    ----------
    X: Optional[SUPPORTED_FEAT_TYPES] = None
        The X data to fit models on, if None, no fitting occurs

    y: Optional[SUPPORTED_FEAT_TYPES] = None
        The y data to fit models on, if None, no fitting occurs

    models: Optional[Collection[AutoSklearnClassificationAlgorithm]] = None
        Any collection of algorithms to use, if None, DummyClassifiers are used
    """

    def _make(
        X: Optional[SUPPORTED_FEAT_TYPES] = None,
        y: Optional[SUPPORTED_TARGET_TYPES] = None,
        models: Optional[Collection[AutoSklearnClassificationAlgorithm]] = None,
        seed: Union[int, None, np.random.RandomState] = DEFAULT_SEED,
    ) -> VotingClassifier:
        assert not (X is None) ^ (y is None)
        if not models:
            validator = InputValidator(is_classification=True).fit(X, y)
            models = [
                MyDummyClassifier(
                    feat_type=validator.feature_validator.feat_type,
                    config=1,
                    random_state=seed,
                )
                for _ in range(5)
            ]

        if X is not None:
            for model in models:
                model.fit(X, y)

        voter = VotingClassifier(estimators=None, voting="soft")
        voter.estimators_ = models
        return voter

    return _make


@fixture
def make_voting_regressor() -> Callable[..., VotingRegressor]:
    """
    Parameters
    ----------
    X: Optional[SUPPORTED_FEAT_TYPES] = None
        The X data to fit models on, if None, no fitting occurs

    y: Optional[SUPPORTED_FEAT_TYPES] = None
        The y data to fit models on, if None, no fitting occurs

    models: Optional[Collection[AutoSklearnRegressionAlgorithm]] = None
        Any collection of algorithms to use, if None, DummyRegressors are used
    """

    def _make(
        X: Optional[SUPPORTED_FEAT_TYPES] = None,
        y: Optional[SUPPORTED_TARGET_TYPES] = None,
        models: Optional[Collection[AutoSklearnRegressionAlgorithm]] = None,
        seed: Union[int, None, np.random.RandomState] = DEFAULT_SEED,
    ) -> VotingRegressor:
        assert not (X is None) ^ (y is None)

        if not models:
            validator = InputValidator(is_classification=False).fit(X, y)
            models = [
                MyDummyRegressor(
                    feat_type=validator.feature_validator.feat_type,
                    config=1,
                    random_state=seed,
                )
                for _ in range(5)
            ]

        if X is not None:
            for model in models:
                model.fit(X, y)

        voter = VotingRegressor(estimators=None)
        voter.estimators_ = models
        return voter

    return _make
