# -*- encoding: utf-8 -*-

import warnings
from typing import List, Tuple, Union

import numpy as np

import pandas as pd

import scipy.sparse

import sklearn.utils
from sklearn import preprocessing
from sklearn.compose import make_column_transformer


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

        # Check for n-outputs change
        self._n_outputs = None

    def validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        is_classification: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for feature/targets validation

        Makes sure consistent number of samples within target and
        features.
        """

        X = self.validate_features(X)
        y = self.validate_target(y, is_classification)

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
        is_classification: bool = False,
    ) -> np.ndarray:
        """
        Wrapper around sklearn check_array. Translates a pandas
        Dataframe to a valid input for sklearn.
        """
        if not hasattr(y, "iloc"):
            y = np.atleast_1d(y)
            if y.ndim == 2 and y.shape[1] == 1:
                warnings.warn("A column-vector y was passed when a 1d array was"
                              " expected. Will change shape via np.ravel().",
                              sklearn.utils.DataConversionWarning, stacklevel=2)
                y = np.ravel(y)

        # During classification, we do ordinal encoding
        # We train a common model for test and train
        # If an encoder was ever done for an estimator,
        # use it always
        # For regression, we default to the check_array in sklearn
        # learn. This handles numerical checking and object conversion
        # For regression, we expect the user to provide numerical input
        # Next check will catch that
        if is_classification or self.target_encoder is not None:
            y = self._check_and_encode_target(y)

        # In code check to make sure everything is numeric
        if hasattr(y, "iloc"):
            is_number = np.vectorize(lambda x: pd.api.types.is_numeric_dtype(x))
            if not np.all(is_number(y.dtypes)):
                raise ValueError(
                    "Input dataframe to autosklearn must only contain numerical"
                    " dtypes, yet it has: {}".format(
                        y.dtypes
                    )
                )
        elif not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                "Input to autosklearn must have a numerical dtype, yet it is: {}".format(
                    y.dtype
                )
            )

        # sklearn check array will make sure we have the
        # correct numerical features for the array
        # Also, a numpy array will be created
        y = sklearn.utils.check_array(
            y,
            force_all_finite=True,
            accept_sparse='csr',
            ensure_2d=False,
        )

        if self._n_outputs is None:
            self._n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
        else:
            _n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
            if self._n_outputs != _n_outputs:
                raise ValueError('Number of outputs changed from %d to %d!' %
                                 (self._n_outputs, _n_outputs))

        return y

    def is_single_column_target(self):
        """
        Output is encoded with a single column encoding
        """
        return self._n_outputs == 1

    def _check_and_get_columns_to_encode(
        self,
        X: pd.DataFrame,
    ) -> Tuple[List[int], List[str]]:
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
                if X[column].dtype.name == 'object':
                    raise ValueError(
                        "Input Column {} has invalid type object. "
                        "Cast it to a valid dtype before using it in Auto-Sklearn. "
                        "Valid types are numerical, categorical or boolean. "
                        "You can cast it to a valid dtype using "
                        "pandas.Series.astype ."
                        "If working with string objects, the following "
                        "tutorial illustrates how to work with text data: "
                        "https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html".format(  # noqa: E501
                            column,
                        )
                    )
                elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    X[column].dtype
                ):
                    raise ValueError(
                        "Auto-sklearn does not support time and/or date datatype as given "
                        "in column {}. Please convert the time information to a numerical value "
                        "first. One example on how to do this can be found on "
                        "https://stats.stackexchange.com/questions/311494/".format(
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
        return enc_columns, feature_types

    def _validate_dataframe_features(
        self,
        X: pd.DataFrame,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Interprets a Pandas
        Uses .iloc as a safe way to deal with pandas object
        """

        # Start with the features
        enc_columns, feature_types = self._check_and_get_columns_to_encode(X)

        # Make sure we only set this once. It should not change
        if not self.feature_types:
            self.feature_types = feature_types

        # This proc has to handle multiple calls, for X_train
        # and X_test scenarios. We have to make sure also that
        # data is consistent within calls
        if enc_columns:
            if self.enc_columns and self.enc_columns != enc_columns:
                raise ValueError(
                    "Changing the column-types of the input data to Auto-Sklearn is not "
                    "allowed. The estimator previously was fitted with categorical/boolean "
                    "columns {}, yet, the new input data has categorical/boolean values {}. "
                    "Please recreate the estimator from scratch when changing the input "
                    "data. ".format(
                        self.enc_columns,
                        enc_columns,
                    )
                )
            else:
                self.enc_columns = enc_columns

            if not self.feature_encoder:
                self.feature_encoder = make_column_transformer(
                    (preprocessing.OrdinalEncoder(), self.enc_columns),
                    remainder="passthrough"
                )

                self.feature_encoder.fit(X)

        if self.feature_encoder:
            X = self.feature_encoder.transform(X)

        # In code check to make sure everything is numeric
        if hasattr(X, "iloc"):
            is_number = np.vectorize(lambda x: pd.api.types.is_numeric_dtype(x))
            if not np.all(is_number(X.dtypes)):
                raise ValueError(
                    "Failed to convert the input dataframe to numerical dtypes: {}".format(
                        X.dtypes
                    )
                )
        elif not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                "Failed to convert the input dataframe to numerical dtype: {}".format(
                    X.dtype
                )
            )

        return X

    def _check_and_encode_target(
        self,
        y: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        This method encodes
        categorical series to a numerical equivalent.

        An ordinal encoder is used for the translation

        """

        # Convert pd.Series to dataframe as categorical series
        # lack many useful methods
        if isinstance(y, pd.Series):
            y = y.to_frame().reset_index(drop=True)

        if hasattr(y, "iloc"):
            self._check_and_get_columns_to_encode(y)

        if not self.target_encoder:
            if y.ndim == 1 or (y.ndim > 1 and y.shape[1] == 1):
                # The label encoder makes sure data is, and remains
                # 1 dimensional
                self.target_encoder = preprocessing.LabelEncoder()
            else:
                self.target_encoder = make_column_transformer(
                    (preprocessing.OrdinalEncoder(), list(range(y.shape[1]))),
                )

            self.target_encoder.fit(y)

        y = self.target_encoder.transform(y)

        return y

    def encode_target(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Encodes the target if there is any encoder
        """
        if self.target_encoder is None:
            return y
        else:

            return self.target_encoder.transform(y)

    def decode_target(
        self,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        If the original target features were encoded,
        this method employs the inverse transform method of the encoder
        to decode the original features
        """
        if self.target_encoder is None:
            return y

        # Handle different ndim encoder for target
        if hasattr(self.target_encoder, 'inverse_transform'):
            return self.target_encoder.inverse_transform(y)
        else:
            return self.target_encoder.named_transformers_['ordinalencoder'].inverse_transform(y)
