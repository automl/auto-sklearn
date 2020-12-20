# -*- encoding: utf-8 -*-
import typing

import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator

from autosklearn.data.feature_validator import FeatureValidator
from autosklearn.data.target_validator import TargetValidator


class InputValidator(BaseEstimator):
    """
    Makes sure the input data complies with Auto-sklearn requirements.
    Categorical inputs are encoded via a Label Encoder, if the input
    is a dataframe.

    This class also perform checks for data integrity and flags the user
    via informative errors.
    """
    def __init__(self) -> None:
        self.feature_validator = FeatureValidator()
        self.target_validator = TargetValidator()
        self._is_fitted = False

    def fit(
        self,
        X_train: typing.Union[np.ndarray, pd.DataFrame],
        y_train: typing.Union[np.ndarray, pd.DataFrame],
        X_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
        y_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
        feat_type: typing.Optional[typing.List[str]] = None,
        is_classification: bool = False,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the features, and
        a encoder for targets in the case of classification.

        Dimensionality and data types checks are also performed

        Parameters
        ----------
            X_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            y_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of targets that are going to be encoded for classification
            X_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of features used for checking
            y_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of targets used for checking
            feat_type: typing.Optional[typing.List[str]]
                In case the data is not a pandas DataFrame, this list indicates
                which columns should be treated as categorical
            is_classification: bool
                For classification task, this flag indicates that the target data
                should be encoded
        """
        # Check that the data is valid
        self.check_data(X_train, y_train, X_test, y_test)
        self.feature_validator.fit(X_train, X_test, feat_type)
        self.target_validator.fit(y_train, y_test, is_classification)
        self._is_fitted = True

        return self

    def transform(
        self,
        X: typing.Union[np.ndarray, pd.DataFrame],
        y: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
    ) -> np.ndarray:
        """
        Transform the given target or features to a numpy array

        Parameters
        ----------
            X: typing.Union[np.ndarray, pd.DataFrame]
                A set of features to transform
            y: typing.Union[np.ndarray, pd.DataFrame]
                A set of targets to transform

        Return
        ------
            np.ndarray:
                The transformed features array
            np.ndarray:
                The transformed targets array
        """
        if not self._is_fitted:
            raise ValueError("Cannot call transform on a validator that is not fitted")
        X_transformed = self.feature_validator.transform(X)
        if y is not None:
            return X_transformed, self.target_validator.transform(y)
        else:
            return X_transformed, y

    def check_data(
        self,
        X_train: typing.Union[np.ndarray, pd.DataFrame],
        y_train: typing.Union[np.ndarray, pd.DataFrame],
        X_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
        y_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
    ) -> None:
        """
        Performs target and feature level checks. Individual checks for features and
        targets are delegated to the respective FeatureValidator() and TargetValidator()

        Parameters
        ----------
            X_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            y_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of targets that are going to be encoded for classification
            X_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of features used for checking
            y_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of targets used for checking
            feat_type: typing.Optional[typing.List[str]]
                In case the data is not a pandas DataFrame, this list indicates
        """
        if np.shape(X_train)[0] != np.shape(y_train)[0]:
            raise ValueError("Inconsistent number of train datapoints for features and targets,"
                             " {} for features and {} for targets".format(
                                 np.shape(X_train)[0],
                                 np.shape(y_train)[0],
                             ))
        if X_test is not None and np.shape(X_test)[0] != np.shape(y_test)[0]:
            raise ValueError("Inconsistent number of test datapoints for features and targets,"
                             " {} for features and {} for targets".format(
                                 np.shape(X_test)[0],
                                 np.shape(y_test)[0],
                             ))
