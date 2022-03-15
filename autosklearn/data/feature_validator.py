from typing import Dict, List, Optional, Tuple, Union, cast

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_sparse
from scipy.sparse import csr_matrix, spmatrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from autosklearn.util.logging_ import PickableLoggerAdapter

SUPPORTED_FEAT_TYPES = Union[List, pd.DataFrame, np.ndarray, spmatrix]


class FeatureValidator(BaseEstimator):
    """
    Checks the input data to Auto-Sklearn.

    It also determines what columns are categorical and which ones are numerical,
    so that the pre-processing pipeline can process this columns accordingly.

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

    data_type:
        Class name of the data type provided during fit.
    """

    def __init__(
        self,
        feat_type: Optional[List[str]] = None,
        logger: Optional[PickableLoggerAdapter] = None,
        allow_string_features: bool = True,
    ) -> None:
        # If a dataframe was provided, we populate
        # this attribute with a mapping from column to
        # {numerical | categorical | string}
        self.feat_type: Optional[Dict[Union[str, int], str]] = None
        if feat_type is not None:
            if isinstance(feat_type, dict):
                self.feat_type = feat_type
            elif not isinstance(feat_type, List):
                raise ValueError(
                    "Auto-Sklearn expects a list of categorical/"
                    "numerical/string feature types, yet a"
                    " {} was provided".format(type(feat_type))
                )
            else:

                # Convert to a dictionary which will be passed to the ColumnTransformer
                # Column Transformer supports strings or integer indexes
                self.feat_type = {i: feat for i, feat in enumerate(feat_type)}

        # Register types to detect unsupported data format changes
        self.data_type = None  # type: Optional[type]
        self.dtypes = {}  # type: Dict[str, str]

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self._is_fitted = False
        self.allow_string_features = allow_string_features

    def fit(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> "FeatureValidator":
        """
        Validates input data to Auto-Sklearn.
        The supported data types are List, numpy arrays and pandas DataFrames.
        CSR sparse data types are also supported

        Parameters
        ----------
        X_train: SUPPORTED_FEAT_TYPES
            A set of features that are going to be validated (type and dimensionality
            checks) and a encoder fitted in the case the data needs encoding
        X_test: Optional[SUPPORTED_FEAT_TYPES]
            A hold out set of data used for checking
        """
        # If a list was provided, it will be converted to pandas
        if isinstance(X_train, List):
            X_train, X_test = self.list_to_dataframe(X_train, X_test)

        self._check_data(X_train)

        # Handle categorical feature identification for the pipeline
        if hasattr(X_train, "iloc"):
            if self.feat_type is not None:
                raise ValueError(
                    "When providing a DataFrame to Auto-Sklearn, we extract "
                    "the feature types from the DataFrame.dtypes. That is, "
                    "providing the option feat_type to the fit method is not "
                    "supported when using a Dataframe. Please make sure that the "
                    "type of each column in your DataFrame is properly set. "
                    "More details about having the correct data type in your "
                    "DataFrame can be seen in "
                    "https://pandas.pydata.org/pandas-docs/stable/reference"
                    "/api/pandas.DataFrame.astype.html"
                )
            else:
                self.feat_type = self.get_feat_type_from_columns(X_train)
        else:
            # Numpy array was provided
            if self.feat_type is None:
                # Assume numerical columns if a numpy array has no feature types
                self.feat_type = {i: "numerical" for i in range(np.shape(X_train)[1])}
            else:
                # Check The feat type provided
                if len(self.feat_type) != np.shape(X_train)[1]:
                    raise ValueError(
                        "Array feat_type does not have same number of "
                        "variables as X has features. %d vs %d."
                        % (len(self.feat_type), np.shape(X_train)[1])
                    )
                if not all([isinstance(f, str) for f in self.feat_type.values()]):
                    raise ValueError(
                        "feat_type must only contain strings: {}".format(
                            list(self.feat_type.values()),
                        )
                    )

                for ft in self.feat_type.values():
                    if ft.lower() not in ["categorical", "numerical", "string"]:
                        raise ValueError(
                            "Only `Categorical`, `Numerical` and `String` are "
                            "valid feature types"
                        )

        if X_test is not None:
            self._check_data(X_test)

            if np.shape(X_train)[1] != np.shape(X_test)[1]:
                raise ValueError(
                    "The feature dimensionality of the train and test "
                    "data does not match train({}) != test({})".format(
                        np.shape(X_train)[1], np.shape(X_test)[1]
                    )
                )

        self._is_fitted = True

        return self

    def transform(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> Union[np.ndarray, spmatrix, pd.DataFrame]:
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
        np.ndarray | spmatrix | pd.DataFrame:
            The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError(
                "Cannot call transform on a validator that is not fitted"
            )

        # If a list was provided, it will be converted to pandas
        if isinstance(X, List):
            X_transformed, _ = self.list_to_dataframe(X)
        else:
            X_transformed = X

        # Check the data here so we catch problems on new test data
        self._check_data(X_transformed)

        # Sparse related transformations
        # Not all sparse format support index sorting
        if isinstance(X_transformed, spmatrix):
            if not isinstance(X_transformed, csr_matrix):
                self.logger.warning(
                    f"Sparse data provided is of type {type(X_transformed)} "
                    "yet Auto-Sklearn only support csr_matrix. Auto-sklearn "
                    "will convert the provided data to the csr_matrix format."
                )
                X_transformed = X_transformed.tocsr(copy=False)

            X_transformed.sort_indices()

        return X_transformed

    def _check_data(
        self,
        X: SUPPORTED_FEAT_TYPES,
    ) -> None:
        """
        Feature dimensionality and data type checks

        Parameters
        ----------
        X: SUPPORTED_FEAT_TYPES
            A set of features that are going to be validated (type and dimensionality)
            and a encoder fitted in the case the data needs encoding
        """
        # We consider columns that are all nan in a pandas frame as category
        if hasattr(X, "columns"):
            for column in cast(pd.DataFrame, X).columns:
                if X[column].isna().all():
                    X[column] = X[column].astype("category")

        if not isinstance(X, (np.ndarray, pd.DataFrame)) and not isinstance(
            X, spmatrix
        ):
            raise ValueError(
                "Auto-sklearn only supports Numpy arrays, Pandas DataFrames,"
                " scipy sparse and Python Lists, yet, the provided input is"
                " of type {}".format(type(X))
            )

        if self.data_type is None:
            self.data_type = type(X)

        if self.data_type != type(X):
            self.logger.warning(
                f"Auto-sklearn previously received features of type {self.data_type} "
                f"yet the current features have type {type(X)}. Changing the dtype "
                "of inputs to an estimator might cause problems"
            )

        # Do not support category/string numpy data. Only numbers
        if hasattr(X, "dtype"):
            if not np.issubdtype(X.dtype.type, np.number):  # type: ignore[union-attr]
                raise ValueError(
                    "When providing a numpy array to Auto-sklearn, the only valid"
                    f" dtypes are numerical ones. The provided data type {X.dtype.type}"
                    " is not supported."
                )

        # Then for Pandas, we do not support Nan in categorical columns
        if hasattr(X, "iloc"):
            # If entered here, we have a pandas dataframe
            X = cast(pd.DataFrame, X)

            dtypes = {col: X[col].dtype.name.lower() for col in X.columns}
            if len(self.dtypes) > 0:
                if self.dtypes != dtypes:
                    # To support list, we need to support object inference.
                    # In extreme cases, the train column might be all integer,
                    # and the test column might be float.
                    self.logger.warning(
                        "Changing the dtype of the features after fit() is "
                        "not recommended. Fit() method was called with "
                        "{} whereas the new features have {} as type".format(
                            self.dtypes,
                            dtypes,
                        )
                    )
            else:
                self.dtypes = dtypes

    def get_feat_type_from_columns(
        self,
        X: pd.DataFrame,
    ) -> Dict[Union[str, int], str]:
        """
        Returns a dictionary that maps pandas dataframe columns to a feature type.
        This feature type can be categorical, numerical or string

        Parameters
        ----------
        X: pd.DataFrame
            A set of features that are going to be validated (type and dimensionality
            checks) and a encoder fitted in the case the data needs encoding

        Returns
        -------
        feat_type:
            dictionary with column to feature type mapping
        """
        # Also, register the feature types for the estimator
        feat_type = {}

        # Make sure each column is a valid type
        for i, column in enumerate(X.columns):
            if is_sparse(X[column]):
                raise ValueError(
                    "Auto-sklearn does not yet support sparse pandas Series."
                    f" Please convert {column} to a dense format."
                )
            elif X[column].dtype.name in ["category", "bool"]:
                feat_type[column] = "categorical"
            elif X[column].dtype.name == "string":
                if self.allow_string_features:
                    feat_type[column] = "string"
                else:
                    feat_type[column] = "categorical"
                    warnings.warn(
                        f"you disabled text encoding column {column} will be "
                        f"encoded as category"
                    )
            # Move away from np.issubdtype as it causes
            # TypeError: data type not understood in certain pandas types
            elif not is_numeric_dtype(X[column]):
                if X[column].dtype.name == "object":
                    warnings.warn(
                        f"Input Column {column} has generic type object. "
                        f"Autosklearn will treat this column as string. "
                        f"Please ensure that this setting is suitable for your task.",
                        UserWarning,
                    )
                    if self.allow_string_features:
                        feat_type[column] = "string"
                    else:
                        feat_type[column] = "categorical"
                        warnings.warn(
                            f"you disabled text encoding column {column} will be"
                            f"encoded as category"
                        )
                elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(
                    X[column].dtype
                ):
                    raise ValueError(
                        "Auto-sklearn does not support time and/or date datatype as "
                        f"given in column {column}. Please convert the time "
                        " information to a numerical value first. One example on how to"
                        " do this can be found on "
                        " https://stats.stackexchange.com/questions/311494/"
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
                feat_type[column] = "numerical"
        return feat_type

    def list_to_dataframe(
        self,
        X_train: SUPPORTED_FEAT_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Converts a list to a DataFrame. In this process, column types are inferred.

        If test data is provided, we proactively match it to train data

        Parameters
        ----------
        X_train: SUPPORTED_FEAT_TYPES
            A set of features that are going to be validated (type and dimensionality
            checks) and a encoder fitted in the case the data needs encoding

        X_test: Optional[SUPPORTED_FEAT_TYPES]
            A hold out set of data used for checking

        Returns
        -------
        Union[pd.DataFrame, pd.DataFrame]:
            transformed (train, test) data from list to pandas DataFrame
        """
        # If a list was provided, it will be converted to pandas
        X_train = pd.DataFrame(data=X_train).convert_dtypes()

        # Store the dtypes and use in case of re-fit
        if len(self.dtypes) == 0:
            self.dtypes = {
                col: X_train[col].dtype.name.lower() for col in X_train.columns
            }
        else:
            for col in X_train.columns:
                # Try to convert to the original dtype used to fit the validator
                # But also be robust to extreme cases (for example, the train data for a
                # column was all np.int-like and the test data is np.float-type)
                try:
                    X_train[col] = X_train[col].astype(self.dtypes[col])
                except Exception as e:
                    self.logger.warning(
                        f"Failed to format column {col} as {self.dtypes[col]}: {e}"
                    )
                    self.dtypes[col] = X_train[col].dtype.name.lower()

        self.logger.warning(
            "The provided feature types to autosklearn are of type list."
            "Features have been interpreted as: {}".format(
                [(col, t) for col, t in zip(X_train.columns, X_train.dtypes)]
            )
        )
        if X_test is not None:
            if not isinstance(X_test, List):
                self.logger.warning(
                    "Train features are a list while the provided test data"
                    "is {}. X_test will be casted as DataFrame.".format(type(X_test))
                )
            X_test = pd.DataFrame(data=X_test)
            for col in X_test.columns:
                try:
                    X_test[col] = X_test[col].astype(self.dtypes[col])
                except Exception as e:
                    self.logger.warning(
                        f"Failed to format column {col} as {self.dtypes[col]}: {e}"
                    )
                    self.dtypes[col] = X_test[col].dtype.name.lower()

        return X_train, X_test
