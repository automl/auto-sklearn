import logging
import typing

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

import scipy.sparse

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target

from autosklearn.util.logging_ import PickableLoggerAdapter


SUPPORTED_TARGET_TYPES = typing.Union[
    typing.List,
    pd.Series,
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


class TargetValidator(BaseEstimator):
    """
    A class to pre-process targets. It validates the data provided during fit (to make sure
    it matches Sklearn expectation) as well as encoding the targets in case of classification
    Attributes
    ----------
        is_classification: bool
            A bool that indicates if the validator should operate in classification mode.
            During classification, the targets are encoded.
        encoder: typing.Optional[BaseEstimator]
            Host a encoder object if the data requires transformation (for example,
            if provided a categorical column in a pandas DataFrame)
        enc_columns: typing.List[str]
            List of columns that where encoded
    """
    def __init__(self,
                 is_classification: bool = False,
                 logger: typing.Optional[PickableLoggerAdapter] = None,
                 ) -> None:
        self.is_classification = is_classification

        self.data_type = None  # type: typing.Optional[type]

        self.encoder = None  # type: typing.Optional[BaseEstimator]

        self.out_dimensionality = None  # type: typing.Optional[int]
        self.type_of_target = None  # type: typing.Optional[str]

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Store the dtype for remapping to correct type
        self.dtype = None  # type: typing.Optional[type]

        self._is_fitted = False

    def fit(
        self,
        y_train: SUPPORTED_TARGET_TYPES,
        y_test: typing.Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the targets
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            y_train: SUPPORTED_TARGET_TYPES
                A set of targets set aside for training
            y_test: typing.Union[SUPPORTED_TARGET_TYPES]
                A hold out set of data used of the targets. It is also used to fit the
                categories of the encoder.
        """
        # Check that the data is valid
        self._check_data(y_train)

        shape = np.shape(y_train)
        if y_test is not None:
            self._check_data(y_test)

            if len(shape) != len(np.shape(y_test)) or (
                    len(shape) > 1 and (shape[1] != np.shape(y_test)[1])):
                raise ValueError("The dimensionality of the train and test targets "
                                 "does not match train({}) != test({})".format(
                                     np.shape(y_train),
                                     np.shape(y_test)
                                 ))
            if isinstance(y_train, pd.DataFrame):
                y_train = typing.cast(pd.DataFrame, y_train)
                y_test = typing.cast(pd.DataFrame, y_test)
                if y_train.columns.tolist() != y_test.columns.tolist():
                    raise ValueError(
                        "Train and test targets must both have the same columns, yet "
                        "y={} and y_test={} ".format(
                            y_train.columns,
                            y_test.columns
                        )
                    )

                if list(y_train.dtypes) != list(y_test.dtypes):
                    raise ValueError("Train and test targets must both have the same dtypes")

        if self.out_dimensionality is None:
            self.out_dimensionality = 1 if len(shape) == 1 else shape[1]
        else:
            _n_outputs = 1 if len(shape) == 1 else shape[1]
            if self.out_dimensionality != _n_outputs:
                raise ValueError('Number of outputs changed from %d to %d!' %
                                 (self.out_dimensionality, _n_outputs))

        # Fit on the training data
        self._fit(y_train, y_test)

        self._is_fitted = True

        return self

    def _fit(
        self,
        y_train: SUPPORTED_TARGET_TYPES,
        y_test: typing.Optional[SUPPORTED_TARGET_TYPES] = None,
    ) -> BaseEstimator:
        """
        If dealing with classification, this utility encodes the targets.

        It does so by also using the classes from the test data, to prevent encoding
        errors

        Parameters
        ----------
            y_train: SUPPORTED_TARGET_TYPES
                The labels of the current task. They are going to be encoded in case
                of classification
            y_test: typing.Optional[SUPPORTED_TARGET_TYPES]
                A holdout set of labels
        """
        if not self.is_classification or self.type_of_target == 'multilabel-indicator':
            # Only fit an encoder for classification tasks
            # Also, encoding multilabel indicator data makes the data multiclass
            # Let the user employ a MultiLabelBinarizer if needed
            return self

        if y_test is not None:
            if hasattr(y_train, "iloc"):
                y_train = pd.concat([y_train, y_test], ignore_index=True, sort=False)
            elif isinstance(y_train, list):
                y_train = y_train + y_test
            elif isinstance(y_train, np.ndarray):
                y_train = np.concatenate((y_train, y_test))

        ndim = len(np.shape(y_train))
        if ndim == 1 or (ndim > 1 and np.shape(y_train)[1] == 1):
            # The label encoder makes sure data is, and remains
            # 1 dimensional
            self.encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                                        unknown_value=-1)
        else:
            # We should not reach this if statement as we check for type of targets before
            raise ValueError("Multi-dimensional classification is not yet supported. "
                             "Encoding multidimensional data converts multiple columns "
                             "to a 1 dimensional encoding. Data involved = {}/{}".format(
                                np.shape(y_train),
                                self.type_of_target
                             ))

        # Mypy redefinition
        assert self.encoder is not None

        # remove ravel warning from pandas Series
        if ndim > 1:
            self.encoder.fit(y_train)
        else:
            if hasattr(y_train, 'iloc'):
                y_train = typing.cast(pd.DataFrame, y_train)
                self.encoder.fit(y_train.to_numpy().reshape(-1, 1))
            else:
                self.encoder.fit(np.array(y_train).reshape(-1, 1))

        # we leave objects unchanged, so no need to store dtype in this case
        if hasattr(y_train, 'dtype'):
            # Series and numpy arrays are checked here
            # Cast is as numpy for mypy checks
            y_train = typing.cast(np.ndarray, y_train)
            if is_numeric_dtype(y_train.dtype):
                self.dtype = y_train.dtype
        elif hasattr(y_train, 'dtypes') and is_numeric_dtype(typing.cast(pd.DataFrame,
                                                                         y_train).dtypes[0]):
            # This case is for pandas array with a single column
            y_train = typing.cast(pd.DataFrame, y_train)
            self.dtype = y_train.dtypes[0]

        return self

    def transform(
        self,
        y: typing.Union[SUPPORTED_TARGET_TYPES],
    ) -> np.ndarray:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            y: SUPPORTED_TARGET_TYPES
                A set of targets that are going to be encoded if the current task
                is classification
        Return
        ------
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call transform on a validator that is not fitted")

        # Check the data here so we catch problems on new test data
        self._check_data(y)

        if self.encoder is not None:
            # remove ravel warning from pandas Series
            shape = np.shape(y)
            if len(shape) > 1:
                y = self.encoder.transform(y)
            else:
                # The Ordinal encoder expects a 2 dimensional input.
                # The targets are 1 dimensional, so reshape to match the expected shape
                if hasattr(y, 'iloc'):
                    y = typing.cast(pd.DataFrame, y)
                    y = self.encoder.transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
                else:
                    y = self.encoder.transform(np.array(y).reshape(-1, 1)).reshape(-1)

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

        return y

    def inverse_transform(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> np.ndarray:
        """
        Revert any encoding transformation done on a target array

        Parameters
        ----------
            y: typing.Union[np.ndarray, pd.DataFrame, pd.Series]
                Target array to be transformed back to original form before encoding
        Return
        ------
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("Cannot call inverse_transform on a validator that is not fitted")

        if self.encoder is None:
            return y
        shape = np.shape(y)
        if len(shape) > 1:
            y = self.encoder.inverse_transform(y)
        else:
            # The targets should be a flattened array, hence reshape with -1
            if hasattr(y, 'iloc'):
                y = typing.cast(pd.DataFrame, y)
                y = self.encoder.inverse_transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
            else:
                y = self.encoder.inverse_transform(np.array(y).reshape(-1, 1)).reshape(-1)

        # Inverse transform returns a numpy array of type object
        # This breaks certain metrics as accuracy, which makes type_of_target be unknown
        # If while fit a dtype was observed, we try to honor that dtype
        if self.dtype is not None:
            y = y.astype(self.dtype)
        return y

    def is_single_column_target(self) -> bool:
        """
        Output is encoded with a single column encoding
        """
        return self.out_dimensionality == 1

    def _check_data(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> None:
        """
        Perform dimensionality and data type checks on the targets

        Parameters
        ----------
            y: typing.Union[np.ndarray, pd.DataFrame, pd.Series]
                A set of features whose dimensionality and data type is going to be checked
        """

        if not isinstance(
                y, (np.ndarray, pd.DataFrame, list, pd.Series)) and not scipy.sparse.issparse(y):
            raise ValueError("Auto-sklearn only supports Numpy arrays, Pandas DataFrames,"
                             " pd.Series, sparse data and Python Lists as targets, yet, "
                             "the provided input is of type {}".format(
                                 type(y)
                             ))

        # Sparse data muss be numerical
        # Type ignore on attribute because sparse targets have a dtype
        if scipy.sparse.issparse(y) and not np.issubdtype(y.dtype.type,  # type: ignore[union-attr]
                                                          np.number):
            raise ValueError("When providing a sparse matrix as targets, the only supported "
                             "values are numerical. Please consider using a dense"
                             " instead."
                             )

        if self.data_type is None:
            self.data_type = type(y)
        if self.data_type != type(y):
            self.logger.warning("Auto-sklearn previously received targets of type %s "
                                "yet the current features have type %s. Changing the dtype "
                                "of inputs to an estimator might cause problems" % (
                                      str(self.data_type),
                                      str(type(y)),
                                   ),
                                )

        # No Nan is supported
        has_nan_values = False
        if hasattr(y, 'iloc'):
            has_nan_values = typing.cast(pd.DataFrame, y).isnull().values.any()
        if scipy.sparse.issparse(y):
            y = typing.cast(scipy.sparse.spmatrix, y)
            has_nan_values = not np.array_equal(y.data, y.data)
        else:
            # List and array like values are considered here
            # np.isnan cannot work on strings, so we have to check for every element
            # but NaN, are not equal to themselves:
            has_nan_values = not np.array_equal(y, y)
        if has_nan_values:
            raise ValueError("Target values cannot contain missing/NaN values. "
                             "This is not supported by scikit-learn. "
                             )

        # Pandas Series is not supported for multi-label indicator
        # This format checks are done by type of target
        try:
            self.type_of_target = type_of_target(y)
        except Exception as e:
            raise ValueError("The provided data could not be interpreted by Sklearn. "
                             "While determining the type of the targets via type_of_target "
                             "run into exception: {}.".format(e))

        supported_output_types = ('binary',
                                  'continuous',
                                  'continuous-multioutput',
                                  'multiclass',
                                  'multilabel-indicator',
                                  # Notice unknown/multiclass-multioutput are not supported
                                  # This can only happen during testing only as estimators
                                  # should filter out unsupported types.
                                  )
        if self.type_of_target not in supported_output_types:
            raise ValueError("Provided targets are not supported by Auto-Sklearn. "
                             "Provided type is {} whereas supported types are {}.".format(
                                 self.type_of_target,
                                 supported_output_types
                             ))

    @property
    def classes_(self) -> np.ndarray:
        """
        Complies with scikit learn classes_ attribute,
        which consist of a ndarray of shape (n_classes,)
        where n_classes are the number of classes seen while fitting
        a encoder to the targets.
        Returns
        -------
            classes_: np.ndarray
                The unique classes seen during encoding of a classifier
        """
        if self.encoder is None:
            return np.array([])
        else:
            return self.encoder.categories_[0]
