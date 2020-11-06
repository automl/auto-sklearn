# -*- encoding: utf-8 -*-

import functools
import warnings
from typing import List, Optional, Tuple, Union

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
    def __init__(self) -> None:
        self.valid_pd_enc_dtypes = ['category', 'bool']

        # If a dataframe was provided, we populate
        # this attribute with the column types from the dataframe
        # That is, this attribute contains whether autosklearn
        # should treat a column as categorical or numerical
        # During fit, if the user provided feature_types, the user
        # constrain is honored. If not, this attribute is used.
        self.feature_types = None  # type: Optional[List[str]]

        # Whereas autosklearn performed encoding on the dataframe
        # We need the target encoder as a decoder mechanism
        self.feature_encoder = None
        self.target_encoder = None
        self.enc_columns = []  # type: List[int]

        # During consecutive calls to the validator,
        # track the number of outputs of the targets
        # We need to make sure y_train/y_test have the
        # same dimensionality
        self._n_outputs = None

        # Add support to make sure that the input to
        # autosklearn has consistent dtype through calls.
        # That is, once fitted, changes in the input dtype
        # are not allowed
        self.features_type = None  # type: Optional[type]
        self.target_type = None  # type: Optional[type]

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

        # Make sure that once fitted, we don't allow new dtypes
        if self.features_type is None:
            self.features_type = type(X)
        if self.features_type != type(X):
            raise ValueError("Auto-sklearn previously received features of type {} "
                             "yet the current features have type {}. Changing the dtype "
                             "of inputs to an estimator is not supported.".format(
                                    self.features_type,
                                    type(X)
                                )
                             )

        # Do not support category/string numpy data. Only numbers
        if hasattr(X, "dtype") and not np.issubdtype(X.dtype.type, np.number):
            raise ValueError(
                "When providing a numpy array to Auto-sklearn, the only valid "
                "dtypes are numerical ones. The provided data type {} is not supported."
                "".format(
                    X.dtype.type,
                )
            )

        # Pre-process dataframe to make them numerical
        # Also, encode numpy categorical objects
        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):
            # Pandas validation provide extra user information
            X = self._check_and_encode_features(X)

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

        # Make sure that once fitted, we don't allow new dtypes
        if self.target_type is None:
            self.target_type = type(y)
        if self.target_type != type(y):
            raise ValueError("Auto-sklearn previously received targets of type {} "
                             "yet the current target has type {}. Changing the dtype "
                             "of inputs to an estimator is not supported.".format(
                                    self.target_type,
                                    type(y)
                                )
                             )

        # Target data as sparse is not supported
        if scipy.sparse.issparse(y):
            raise ValueError("Unsupported target data provided"
                             "Input targets to auto-sklearn must not be of "
                             "type sparse. Please convert the target input (y) "
                             "to a dense array via scipy.sparse.csr_matrix.todense(). "
                             )

        # No Nan is supported
        if np.any(pd.isnull(y)):
            raise ValueError("Target values cannot contain missing/NaN values. "
                             "This is not supported by scikit-learn. "
                             )

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
                    "During the target validation (y_train/y_test) an invalid"
                    " input was detected. "
                    "Input dataframe to autosklearn must only contain numerical"
                    " dtypes, yet it has: {} dtypes.".format(
                        y.dtypes
                    )
                )
        elif not np.issubdtype(y.dtype, np.number):
            raise ValueError(
                "During the target validation (y_train/y_test) an invalid"
                " input was detected. "
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

        # When translating a dataframe to numpy, make sure we
        # honor the ravel requirement
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        if self._n_outputs is None:
            self._n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
        else:
            _n_outputs = 1 if len(y.shape) == 1 else y.shape[1]
            if self._n_outputs != _n_outputs:
                raise ValueError('Number of outputs changed from %d to %d!' %
                                 (self._n_outputs, _n_outputs))

        return y

    def is_single_column_target(self) -> bool:
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

                if hasattr(X, "iloc"):
                    enc_columns.append(column)
                else:
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

    def _check_and_encode_features(
        self,
        X: pd.DataFrame,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Interprets a Pandas
        Uses .iloc as a safe way to deal with pandas object
        """

        # If there is a Nan, we cannot encode it due to a scikit learn limitation
        if np.any(pd.isnull(X.dropna(axis='columns', how='all'))):
            # Ignore all NaN columns, and if still a NaN
            # Error out
            raise ValueError("Categorical features in a dataframe cannot contain "
                             "missing/NaN values. The OrdinalEncoder used by "
                             "Auto-sklearn cannot handle this yet (due to a "
                             "limitation on scikit-learn being addressed via: "
                             "https://github.com/scikit-learn/scikit-learn/issues/17123)"
                             )
        elif np.any(pd.isnull(X)):
            # After above check it means that if there is a NaN
            # the whole column must be NaN
            # Make sure it is numerical and let the pipeline handle it
            for column in X.columns:
                if X[column].isna().all():
                    X[column] = pd.to_numeric(X[column])

        # Start with the features
        if hasattr(X, "iloc"):
            enc_columns, feature_types = self._check_and_get_columns_to_encode(X)
        else:
            if len(X.shape) < 1:
                raise ValueError("Input data X cannot be one dimensional "
                                 "and need to be reshaped to a 2-D array-like object."
                                 "You can do so via np.reshape(-1,1). "
                                 )
            enc_columns = list(range(X.shape[1]))
            feature_types = ['categorical' for f in enc_columns]

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

                # Mypy redefinition
                assert self.feature_encoder is not None
                self.feature_encoder.fit(X)

                # The column transformer reoders the feature types - we therefore need to change
                # it as well
                def comparator(cmp1, cmp2):
                    if (
                        cmp1 == 'categorical' and cmp2 == 'categorical'
                        or cmp1 == 'numerical' and cmp2 == 'numerical'
                    ):
                        return 0
                    elif cmp1 == 'categorical' and cmp2 == 'numerical':
                        return -1
                    elif cmp1 == 'numerical' and cmp2 == 'categorical':
                        return 1
                    else:
                        raise ValueError((cmp1, cmp2))
                self.feature_types = sorted(
                    self.feature_types,
                    key=functools.cmp_to_key(comparator)
                )

        if self.feature_encoder:
            try:
                X = self.feature_encoder.transform(X)
            except ValueError as e:
                if 'Found unknown categories' in e.args[0]:
                    # Make the message more informative
                    raise ValueError(
                        "During fit, the input features contained categorical values in columns"
                        "{}, with categories {} which were encoded by Auto-sklearn automatically."
                        "Nevertheless, a new input contained new categories not seen during "
                        "training = {}. The OrdinalEncoder used by Auto-sklearn cannot handle "
                        "this yet (due to a limitation on scikit-learn being addressed via:"
                        " https://github.com/scikit-learn/scikit-learn/issues/17123)"
                        "".format(
                            self.enc_columns,
                            self.feature_encoder.transformers_[0][1].categories_,
                            e.args[0],
                        )
                    )
                else:
                    raise e

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
                "Failed to convert the input array to numerical dtype: {}".format(
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

            # Mypy redefinition
            assert self.target_encoder is not None

            # remove ravel warning from pandas Series
            if len(y.shape) > 1 and y.shape[1] == 1 and hasattr(y, "to_numpy"):
                self.target_encoder.fit(y.to_numpy().ravel())
            else:
                self.target_encoder.fit(y)

        try:
            # remove ravel warning from pandas Series
            if len(y.shape) > 1 and y.shape[1] == 1 and hasattr(y, "to_numpy"):
                y = self.target_encoder.transform(y.to_numpy().ravel())
            else:
                y = self.target_encoder.transform(y)
        except ValueError as e:
            if 'Found unknown categories' in e.args[0]:
                # Make the message more informative for Ordinal
                raise ValueError(
                    "During fit, the target array contained the categorical values {} "
                    "which were encoded by Auto-sklearn automatically. "
                    "Nevertheless, a new target set contained new categories not seen during "
                    "training = {}. The OrdinalEncoder used by Auto-sklearn cannot handle "
                    "this yet (due to a limitation on scikit-learn being addressed via:"
                    " https://github.com/scikit-learn/scikit-learn/issues/17123)"
                    "".format(
                        self.target_encoder.transformers_[0][1].categories_,
                        e.args[0],
                    )
                )
            elif 'contains previously unseen labels' in e.args[0]:
                # Make the message more informative
                raise ValueError(
                    "During fit, the target array contained the categorical values {} "
                    "which were encoded by Auto-sklearn automatically. "
                    "Nevertheless, a new target set contained new categories not seen during "
                    "training = {}. This is a limitation in scikit-learn encoders being "
                    "discussed in //github.com/scikit-learn/scikit-learn/issues/17123".format(
                        self.target_encoder.classes_,
                        e.args[0],
                    )
                )
            else:
                raise e

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

    def join_and_check(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        This method checks for basic input quality before
        merging the inputs to Auto-sklearn for a common
        encoding if needed
        """

        # We expect same type of object
        if type(y) != type(y_test):
            raise ValueError(
                "Train and test targets must be of the same type, yet y={} and y_test={}"
                "".format(
                    type(y),
                    type(y_test)
                )
            )

        if isinstance(y, pd.DataFrame):
            # The have to have the same columns and types
            if y.columns != y_test.columns:
                raise ValueError(
                    "Train and test targets must both have the same columns, yet "
                    "y={} and y_test={} ".format(
                        type(y),
                        type(y_test)
                    )
                )

            if list(y.dtypes) != list(y_test.dtypes):
                raise ValueError("Train and test targets must both have the same dtypes")

            return pd.concat([y, y_test], ignore_index=True, sort=False)
        elif isinstance(y, np.ndarray):
            # The have to have the same columns and types
            if len(y.shape) != len(y_test.shape) \
                    or (len(y.shape) > 1 and (y.shape[1] != y_test.shape[1])):
                raise ValueError("Train and test targets must have the same dimensionality")

            if y.dtype != y_test.dtype:
                raise ValueError("Train and test targets must both have the same dtype")

            return np.concatenate((y, y_test))
        elif isinstance(y, list):
            # Provide flexibility in the list. When transformed to np.ndarray
            # further checks are performed downstream
            return y + y_test
        elif scipy.sparse.issparse(y):
            # Here just return y, vstack from scipy cause ufunc 'isnan' type errors
            # in multilabel sparse matrices. Since we don't encode scipy matrices,
            # No functionality impact.
            return y
        else:
            raise ValueError("Unsupported input type y={type(y)}. Auto-Sklearn supports "
                             "Pandas DataFrames, numpy arrays, scipy csr  or  python lists. "
                             "Kindly cast your targets to a supported type."
                             )
