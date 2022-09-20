# -*- encoding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import logging

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autosklearn.data.feature_validator import SUPPORTED_FEAT_TYPES, FeatureValidator
from autosklearn.data.target_validator import SUPPORTED_TARGET_TYPES, TargetValidator
from autosklearn.util.logging_ import get_named_client_logger


def convert_if_sparse(
    y: SUPPORTED_TARGET_TYPES,
) -> Union[np.ndarray, List, pd.DataFrame, pd.Series]:
    """If the labels `y` are sparse, it will convert it to its dense representation

    Parameters
    ----------
    y: {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
        The labels to 'densify' if sparse

    Returns
    -------
    np.ndarray of shape (n_samples, ) or (n_samples, n_outputs)
    """
    if isinstance(y, spmatrix):
        y_ = y.toarray()

        # For sparse one dimensional data, y.toarray will return [[1], [2], [3], ...]
        # We need to flatten this before returning it
        if y_.shape[0] == 1:
            y_ = y_.flatten()
    else:
        y_ = y

    return y_


class InputValidator(BaseEstimator):
    """
    Makes sure the input data complies with Auto-sklearn requirements.
    Categorical inputs are encoded via a Label Encoder, if the input
    is a dataframe.

    This class also perform checks for data integrity and flags the user
    via informative errors.

    Attributes
    ----------
    feat_type: Optional[List[str]] = None
        In case the dataset is not a pandas DataFrame:
            +   If provided, this list indicates which columns should be treated as
                categorical it is internally transformed into a dictionary that
                indicates a mapping from column index to categorical/numerical.
            +   If not provided, by default all columns are treated as numerical

        If the input dataset is of type pandas dataframe, this argument
        must be none, as the column type will be inferred from the pandas dtypes.

    is_classification: bool
        For classification task, this flag indicates that the target data
        should be encoded

    feature_validator: FeatureValidator
        A FeatureValidator instance used to validate and encode feature columns to match
        sklearn expectations on the data

    target_validator: TargetValidator
        A TargetValidator instance used for classification to validate and encode the
        target values
    """

    def __init__(
        self,
        feat_type: Optional[List[str]] = None,
        is_classification: bool = False,
        logger_port: Optional[int] = None,
        allow_string_features: bool = True,
    ) -> None:
        self.feat_type = feat_type
        self.is_classification = is_classification
        self.logger_port = logger_port
        if self.logger_port is not None:
            self.logger = get_named_client_logger(
                name="Validation",
                port=self.logger_port,
            )
        else:
            self.logger = logging.getLogger("Validation")

        self.allow_string_features = allow_string_features
        self.feature_validator = FeatureValidator(
            feat_type=self.feat_type,
            logger=self.logger,
            allow_string_features=self.allow_string_features,
        )
        self.target_validator = TargetValidator(
            is_classification=self.is_classification, logger=self.logger
        )
        self._is_fitted = False

    def fit(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        y_train: SUPPORTED_TARGET_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
        y_test: Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the features, and
        a encoder for targets in the case of classification. Specifically:

        For features:
        Valid data types are enforced (List, np.ndarray, pd.DataFrame, pd.Series, scipy
        sparse) as well as dimensionality checks

        If the provided data is a pandas DataFrame with categorical/boolean/int columns,
        such columns will be encoded using an Ordinal Encoder

        For targets:
        * Checks for dimensionality as well as missing values are performed.
        * If performing a classification task, the data is going to be encoded

        Parameters
        ----------
        X_train: SUPPORTED_FEAT_TYPES
            A set of features that are going to be validated (type and dimensionality
            checks). If this data contains categorical columns, an encoder is going to
            be instantiated and trained with this data.

        y_train: SUPPORTED_TARGET_TYPES
            A set of targets to encoded if the task is for classification.

        X_test: Optional[SUPPORTED_FEAT_TYPES]
            A hold out set of features used for checking

        y_test: SUPPORTED_TARGET_TYPES
            A hold out set of targets used for checking. Additionally, if the current
            task is a classification task, this y_test categories are also going to be
            used to fit a pre-processing encoding (to prevent errors on unseen classes).

        Returns
        -------
        self
        """
        # Check that the data is valid
        if np.shape(X_train)[0] != np.shape(y_train)[0]:
            raise ValueError(
                "Inconsistent number of train datapoints for features and targets,"
                " {} for features and {} for targets".format(
                    np.shape(X_train)[0],
                    np.shape(y_train)[0],
                )
            )
        if X_test is not None and np.shape(X_test)[0] != np.shape(y_test)[0]:
            raise ValueError(
                "Inconsistent number of test datapoints for features and targets,"
                " {} for features and {} for targets".format(
                    np.shape(X_test)[0],
                    np.shape(y_test)[0],
                )
            )

        self.feature_validator.fit(X_train, X_test)
        self.target_validator.fit(y_train, y_test)
        self._is_fitted = True

        return self

    def transform(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: Optional[Union[List, pd.Series, pd.DataFrame, np.ndarray]] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame, spmatrix], Optional[np.ndarray]]:
        """
        Transform the given target or features to a numpy array

        Parameters
        ----------
            X: SUPPORTED_FEAT_TYPES
                A set of features to transform
            y: Optional[SUPPORTED_TARGET_TYPES]
                A set of targets to transform

        Return
        ------
            np.ndarray:
                The transformed features array
            np.ndarray:
                The transformed targets array
        """
        if not self._is_fitted:
            raise NotFittedError(
                "Cannot call transform on a validator that is not fitted"
            )

        X_transformed = self.feature_validator.transform(X)
        if y is not None:
            y_transformed = self.target_validator.transform(y)
            return X_transformed, y_transformed
        else:
            return X_transformed, None
