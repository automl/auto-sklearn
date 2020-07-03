# -*- encoding: utf-8 -*-

from typing import Union, Tuple
import warnings

import numpy as np
import pandas as pd

import sklearn.utils
from sklearn import preprocessing
from sklearn.compose import make_column_transformer

import scipy.sparse


class InputValidator:
    """
    Makes sure the input data complies with Auto-sklearn requirements.
    Categorical inputs are encoded via a Label Encoder, if the input
    is a dataframe.

    This class also perform checks for data integrity and flags the user
    via informative errors.
    """
    def __init__(
        self,
        feature_encoder_type: str = 'OrdinalEncoder',
        target_encoder_type: str = 'LabelEncoder',
    ):
        self.valid_pd_enc_dtypes = ['category', 'bool']

        # Whether we should treat the feature as a
        # categorical or numerical input. If None,
        # (The input was not a dataframe)
        # The estimator will try to guess
        self.feature_types = None

        # Whereas autosklearn performed encoding on the dataframe
        # We need the target encoder as a decoder mechanism
        self.feature_encoder = None
        self.target_encoder = None
        self.enc_columns = []

        self.feature_encoder_type = feature_encoder_type
        self.target_encoder_type = target_encoder_type

    def validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for feature/targets validation

        Makes sure consistent number of samples within target and
        features.
        """

        X = self.validate_features(X)
        y = self.validate_target(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples from the features X={} should match "
                "the number of samples from the target y={}".format(
                    X.shape[0],
                    y.shape[0]
                )
            )
        return X, y

    def validate_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Wrapper around sklearn check_array. Translates a pandas
        Dataframe to a valid input for sklearn.
        """

        # Pre-process dataframe to make them numerical
        if hasattr(X, "iloc"):
            # Pandas validation provide extra user information
            X = self._validate_dataframe_features(X)

        if scipy.sparse.issparse(X):
            X.sort_indices()

        # sklearn check array will make sure we have the
        # correct numerical features for the array
        # Also, a numpy array will be created
        X = sklearn.utils.check_array(
            X,
            force_all_finite=False,
            accept_sparse='csr'
        )
        return X

    def validate_target(
        self,
        y: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Wrapper around sklearn check_array. Translates a pandas
        Dataframe to a valid input for sklearn.
        """
        # Support also Pandas object in the targets
        if hasattr(y, "iloc"):
            y = self._validate_dataframe_target(y)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Will change shape via np.ravel().",
                          sklearn.utils.DataConversionWarning, stacklevel=2)
            y = np.ravel(y)

        # sklearn check array will make sure we have the
        # correct numerical features for the array
        # Also, a numpy array will be created
        y = sklearn.utils.check_array(
            y,
            force_all_finite=True,
            accept_sparse='csr',
            ensure_2d=False,
        )
        return y

    def _validate_dataframe_features(
        self,
        X: pd.DataFrame,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Interprets a Pandas
        Uses .iloc as a safe way to deal with pandas object
        """

        # Start with the features

        # Register if a column needs encoding
        enc_columns = []

        # Also, register the feature types for the estimator
        feature_types = []

        # Make sure each column is a valid type
        for i, column in enumerate(X.columns):
            if X[column].dtype.name in self.valid_pd_enc_dtypes:
                enc_columns.append(i)
                feature_types.append('categorical')
            elif not np.issubdtype(X[column].dtype, np.number):
                if X[column].dtype.name == 'Object':
                    raise ValueError(
                        "Input Column {} has invalid type object. "
                        "Cast it to a numerical value before using it in Auto-Sklearn."
                        "You can use pandas.to_numeric() to do this".format(
                            column,
                        )
                    )
                elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    X[column].dtype
                ):
                    raise ValueError(
                        "Input Column {} has invalid time-type. "
                        "Please convert the time information to a categorical value. "
                        "Usually, time information can be separated into a set of categorical "
                        "features. For example, hour of the day can be converted to 24 "
                        " categories.".format(
                            column,
                        )
                    )
                else:
                    raise ValueError(
                        "Input Column {} has unsupported dtype {}. "
                        "Supported column types are categorical/bool/numerical dtypes. "
                        "Make sure your data is formatted in a correct way, "
                        "before feeding it to Auto-Sklearn.".format(
                            column,
                            X[column].dtype.name,
                        )
                    )
            else:
                feature_types.append('numerical')

        # Make sure we only set this once. It should not change
        if not self.feature_types:
            self.feature_types = feature_types

        # This proc has to handle multiple calls, for X_train
        # and X_test scenarios. We have to make sure also that
        # data is consistent within calls
        if enc_columns:
            if self.enc_columns and self.enc_columns != enc_columns:
                raise ValueError(
                    "Validation was already performed on input with enc columns={} "
                    "yet, another set of features with columns to encode={} was "
                    "provided. Feature changing after fit is not allowed.".format(
                        enc_columns,
                        self.enc_columns
                    )
                )
            else:
                self.enc_columns = enc_columns

            if not self.feature_encoder:
                if self.feature_encoder_type == 'OneHotEncoder':
                    encoder = preprocessing.OneHotEncoder(sparse=False)
                elif self.feature_encoder_type == 'OrdinalEncoder':
                    encoder = preprocessing.OrdinalEncoder()
                else:
                    raise ValueError(
                        "Unsupported feature encoder provided {}".format(
                            self.feature_encoder_type
                        )
                    )

                self.feature_encoder = make_column_transformer(
                    (encoder, self.enc_columns),
                    remainder="passthrough"
                )

                self.feature_encoder.fit(X)

        if self.feature_encoder:
            X = self.feature_encoder.transform(X)

        return X

    def _validate_dataframe_target(
        self,
        y: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        This method encodes
        categorical series to a numerical equivalent.

        No change to numerical values.

        Dataframes with more than 1 column are ambiguous because
        two column could be a label or multilabel. Force the user
        to use numerical encoding via sklearn checks
        """

        # Convert pd.Series to dataframe as categorical series
        # lack many useful methods
        if isinstance(y, pd.Series):
            y = y.to_frame().reset_index(drop=True)

        # handle categorical encoding of a Dataframes
        # with a single column
        # A dataframe target with multiple columns is ambiguous
        # We don't know if all columns should share same encoding
        # or Not

        if y.ndim == 2 and y.shape[1] == 1:
            if y.dtypes[0].name in self.valid_pd_enc_dtypes:
                if self.target_encoder_type == 'OneHotEncoder':
                    self.target_encoder = preprocessing.OneHotEncoder(sparse=False)
                elif self.target_encoder_type == 'OrdinalEncoder':
                    self.target_encoder = preprocessing.OrdinalEncoder()
                elif self.target_encoder_type == 'LabelEncoder':
                    self.target_encoder = preprocessing.LabelEncoder()
                else:
                    raise ValueError(
                        "Unsupported feature encoder provided {}".format(
                            self.target_encoder_type
                        )
                    )
                self.target_encoder.fit(y.values.reshape(-1, 1))
            elif not np.issubdtype(y.dtypes[0], np.number):
                raise ValueError(
                    "Target Series y has unsupported dtype {}. "
                    "Supported column types are categorical/bool/numerical dtypes. "
                    "Make sure your data is formatted in a correct way, "
                    "before feeding it to Auto-Sklearn.".format(
                        y.dtypes[0].name,
                    )
                )
            if self.target_encoder:
                y = self.target_encoder.transform(y)

        return y

    def decode(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        If the original target features were encoded,
        this method employs the inverse transform method of the encoder
        to decode the original features
        """
        if self.target_encoder is None:
            raise ValueError(
                "No point in calling decode when the input y was not encoded"
            )
        return self.target_encoder.inverse_transform(y)
