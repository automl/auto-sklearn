import functools
import logging
import typing

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

import scipy.sparse

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.exceptions import NotFittedError

from autosklearn.util.logging_ import PickableLoggerAdapter


SUPPORTED_FEAT_TYPES = typing.Union[
    typing.List,
    pd.DataFrame,
    np.ndarray,
    scipy.sparse.bsr_matrix,
    scipy.sparse.coo_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.dia_matrix,
    scipy.sparse.dok_matrix,
    scipy.sparse.lil_matrix,
]


class FeatureValidator(BaseEstimator):
    """
    A class to pre-process features. In this regards, the format of the data is checked,
    and if applicable, features are encoded
    Attributes
    ----------
        feat_type: typing.Optional[typing.List[str]]
            In case the data is not a pandas DataFrame, this list indicates
            which columns should be treated as categorical
        data_type:
            Class name of the data type provided during fit.
        encoder: typing.Optional[BaseEstimator]
            Host a encoder object if the data requires transformation (for example,
            if provided a categorical column in a pandas DataFrame)
        enc_columns: typing.List[str]
            List of columns that where encoded
    """
    def __init__(self,
                 feat_type: typing.Optional[typing.List[str]] = None,
                 logger: typing.Optional[PickableLoggerAdapter] = None,
                 ) -> None:
        # If a dataframe was provided, we populate
        # this attribute with the column types from the dataframe
        # That is, this attribute contains whether autosklearn
        # should treat a column as categorical or numerical
        # During fit, if the user provided feat_types, the user
        # constrain is honored. If not, this attribute is used.
        self.feat_type = feat_type  # type: typing.Optional[typing.List[str]]

        # Register types to detect unsupported data format changes
        self.data_type = None  # type: typing.Optional[type]
        self.dtypes = []  # type: typing.List[str]
        self.column_order = []  # type: typing.List[str]

        self.encoder = None  # type: typing.Optional[BaseEstimator]
        self.enc_columns = []  # type: typing.List[str]

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._is_fitted = False

    def fit(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: typing.Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.
        CSR sparse data types are also supported

        Parameters
        ----------
            X_train: SUPPORTED_FEAT_TYPES
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test: typing.Optional[SUPPORTED_FEAT_TYPES]
                A hold out set of data used for checking
        """

        # If a list was provided, it will be converted to pandas
        if isinstance(X_train, list):
            X_train, X_test = self.list_to_dataframe(X_train, X_test)

        # Register the user provided feature types
        if self.feat_type is not None:
            if hasattr(X_train, "iloc"):
                raise ValueError("When providing a DataFrame to Auto-Sklearn, we extract "
                                 "the feature types from the DataFrame.dtypes. That is, "
                                 "providing the option feat_type to the fit method is not "
                                 "supported when using a Dataframe. Please make sure that the "
                                 "type of each column in your DataFrame is properly set. "
                                 "More details about having the correct data type in your "
                                 "DataFrame can be seen in "
                                 "https://pandas.pydata.org/pandas-docs/stable/reference"
                                 "/api/pandas.DataFrame.astype.html")
            # Some checks if self.feat_type is provided
            if len(self.feat_type) != np.shape(X_train)[1]:
                raise ValueError('Array feat_type does not have same number of '
                                 'variables as X has features. %d vs %d.' %
                                 (len(self.feat_type), np.shape(X_train)[1]))
            if not all([isinstance(f, str) for f in self.feat_type]):
                raise ValueError('Array feat_type must only contain strings.')

            for ft in self.feat_type:
                if ft.lower() not in ['categorical', 'numerical']:
                    raise ValueError('Only `Categorical` and `Numerical` are '
                                     'valid feature types, you passed `%s`' % ft)

        self._check_data(X_train)

        if X_test is not None:
            self._check_data(X_test)

            if np.shape(X_train)[1] != np.shape(X_test)[1]:
                raise ValueError("The feature dimensionality of the train and test "
                                 "data does not match train({}) != test({})".format(
                                     np.shape(X_train)[1],
                                     np.shape(X_test)[1]
                                 ))

        # Fit on the training data
        self._fit(X_train)

        self._is_fitted = True

        return self

    def _fit(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> BaseEstimator:
        """
        In case input data is a pandas DataFrame, this utility encodes the user provided
        features (from categorical for example) to a numerical value that further stages
        will be able to use

        Parameters
        ----------
            X: SUPPORTED_FEAT_TYPES
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        """
        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):
            X = typing.cast(pd.DataFrame, X)
            # Treat a column with all instances a NaN as numerical
            # This will prevent doing encoding to a categorical column made completely
            # out of nan values -- which will trigger a fail, as encoding is not supported
            # with nan values.
            # Columns that are completely made of NaN values are provided to the pipeline
            # so that later stages decide how to handle them
            if np.any(pd.isnull(X)):
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])

            self.enc_columns, self.feat_type = self._get_columns_to_encode(X)

            if len(self.enc_columns) > 0:

                self.encoder = make_column_transformer(
                    (preprocessing.OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                    ), self.enc_columns),
                    remainder="passthrough"
                )

                # Mypy redefinition
                assert self.encoder is not None
                self.encoder.fit(X)

                # The column transformer reoders the feature types - we therefore need to change
                # it as well
                def comparator(cmp1: str, cmp2: str) -> int:
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
                self.feat_type = sorted(
                    self.feat_type,
                    key=functools.cmp_to_key(comparator)
                )
        return self

    def transform(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> np.ndarray:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            X_train: SUPPORTED_FEAT_TYPES
                A set of features, whose categorical features are going to be
                transformed

        Return
        ------
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # If a list was provided, it will be converted to pandas
        if isinstance(X, list):
            X, _ = self.list_to_dataframe(X)

        if hasattr(X, "iloc") and not scipy.sparse.issparse(X):
            X = typing.cast(pd.DataFrame, X)
            if np.any(pd.isnull(X)):
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])

        # Check the data here so we catch problems on new test data
        self._check_data(X)

        # Pandas related transformations
        if hasattr(X, "iloc") and self.encoder is not None:
            if np.any(pd.isnull(X)):
                # After above check it means that if there is a NaN
                # the whole column must be NaN
                # Make sure it is numerical and let the pipeline handle it
                for column in X.columns:
                    if X[column].isna().all():
                        X[column] = pd.to_numeric(X[column])
            X = self.encoder.transform(X)

        # Sparse related transformations
        # Not all sparse format support index sorting
        if scipy.sparse.issparse(X) and hasattr(X, 'sort_indices'):
            X.sort_indices()

        return sklearn.utils.check_array(
            X,
            force_all_finite=False,
            accept_sparse='csr'
        )

    def _check_data(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> None:
        """
        Feature dimensionality and data type checks

        Parameters
        ----------
            X: SUPPORTED_FEAT_TYPES
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        """

        if not isinstance(X, (np.ndarray, pd.DataFrame)) and not scipy.sparse.issparse(X):
            raise ValueError("Auto-sklearn only supports Numpy arrays, Pandas DataFrames,"
                             " scipy sparse and Python Lists, yet, the provided input is"
                             " of type {}".format(
                                 type(X)
                             ))

        if self.data_type is None:
            self.data_type = type(X)
        if self.data_type != type(X):
            self.logger.warning("Auto-sklearn previously received features of type %s "
                                "yet the current features have type %s. Changing the dtype "
                                "of inputs to an estimator might cause problems" % (
                                      str(self.data_type),
                                      str(type(X)),
                                   ),
                                )

        # Do not support category/string numpy data. Only numbers
        if hasattr(X, "dtype"):
            if not np.issubdtype(X.dtype.type, np.number):  # type: ignore[union-attr]
                raise ValueError(
                    "When providing a numpy array to Auto-sklearn, the only valid "
                    "dtypes are numerical ones. The provided data type {} is not supported."
                    "".format(
                        X.dtype.type,  # type: ignore[union-attr]
                    )
                )

        # Then for Pandas, we do not support Nan in categorical columns
        if hasattr(X, "iloc"):
            # If entered here, we have a pandas dataframe
            X = typing.cast(pd.DataFrame, X)

            # Define the column to be encoded here as the feature validator is fitted once
            # per estimator
            enc_columns, _ = self._get_columns_to_encode(X)

            if len(enc_columns) > 0:
                if np.any(pd.isnull(
                    X[enc_columns].dropna(  # type: ignore[call-overload]
                        axis='columns', how='all')
                )):
                    # Ignore all NaN columns, and if still a NaN
                    # Error out
                    raise ValueError("Categorical features in a dataframe cannot contain "
                                     "missing/NaN values. The OrdinalEncoder used by "
                                     "Auto-sklearn cannot handle this yet (due to a "
                                     "limitation on scikit-learn being addressed via: "
                                     "https://github.com/scikit-learn/scikit-learn/issues/17123)"
                                     )
            column_order = [column for column in X.columns]
            if len(self.column_order) > 0:
                if self.column_order != column_order:
                    raise ValueError("Changing the column order of the features after fit() is "
                                     "not supported. Fit() method was called with "
                                     "{} whereas the new features have {} as type".format(
                                        self.column_order,
                                        column_order,
                                     ))
            else:
                self.column_order = column_order
            dtypes = [dtype.name for dtype in X.dtypes]
            if len(self.dtypes) > 0:
                if self.dtypes != dtypes:
                    raise ValueError("Changing the dtype of the features after fit() is "
                                     "not supported. Fit() method was called with "
                                     "{} whereas the new features have {} as type".format(
                                        self.dtypes,
                                        dtypes,
                                     ))
            else:
                self.dtypes = dtypes

    def _get_columns_to_encode(
        self,
        X: pd.DataFrame,
    ) -> typing.Tuple[typing.List[str], typing.List[str]]:
        """
        Return the columns to be encoded from a pandas dataframe

        Parameters
        ----------
            X: pd.DataFrame
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
        Returns
        -------
            enc_columns:
                Columns to encode, if any
            feat_type:
                Type of each column numerical/categorical
        """
        # Register if a column needs encoding
        enc_columns = []

        # Also, register the feature types for the estimator
        feat_type = []

        # Make sure each column is a valid type
        for i, column in enumerate(X.columns):
            if X[column].dtype.name in ['category', 'bool']:

                enc_columns.append(column)
                feat_type.append('categorical')
            # Move away from np.issubdtype as it causes
            # TypeError: data type not understood in certain pandas types
            elif not is_numeric_dtype(X[column]):
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
                feat_type.append('numerical')
        return enc_columns, feat_type

    def list_to_dataframe(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: typing.Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> typing.Tuple[pd.DataFrame, typing.Optional[pd.DataFrame]]:
        """
        Converts a list to a pandas DataFrame. In this process, column types are inferred.

        If test data is provided, we proactively match it to train data

        Parameters
        ----------
            X_train: SUPPORTED_FEAT_TYPES
                A set of features that are going to be validated (type and dimensionality
                checks) and a encoder fitted in the case the data needs encoding
            X_test: typing.Optional[SUPPORTED_FEAT_TYPES]
                A hold out set of data used for checking
        Returns
        -------
            pd.DataFrame:
                transformed train data from list to pandas DataFrame
            pd.DataFrame:
                transformed test data from list to pandas DataFrame
        """

        # If a list was provided, it will be converted to pandas
        X_train = pd.DataFrame(data=X_train).infer_objects()
        self.logger.warning("The provided feature types to autosklearn are of type list."
                            "Features have been interpreted as: {}".format(
                                [(col, t) for col, t in zip(X_train.columns, X_train.dtypes)]
                            ))
        if X_test is not None:
            if not isinstance(X_test, list):
                self.logger.warning("Train features are a list while the provided test data"
                                    "is {}. X_test will be casted as DataFrame.".format(
                                        type(X_test)
                                    ))
            X_test = pd.DataFrame(data=X_test).infer_objects()
        return X_train, X_test
