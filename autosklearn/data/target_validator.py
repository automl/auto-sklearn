from typing import List, Optional, Type, Union, cast

import logging
import warnings

import numpy as np
import pandas as pd
import sklearn.utils
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.base import ExtensionDtype
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import type_of_target

from autosklearn.util.logging_ import PickableLoggerAdapter

SUPPORTED_TARGET_TYPES = Union[List, pd.Series, pd.DataFrame, np.ndarray, spmatrix]
SUPPORTED_XDATA_TYPES = Union[pd.Series, pd.DataFrame, np.ndarray, spmatrix]


class TargetValidator(BaseEstimator):
    """A class to pre-process targets.

    It validates the data provided during fit to make sure it matches Sklearn
    expectation as well as encoding the targets in case of classification

    Attributes
    ----------
    is_classification: bool
        A bool that indicates if the validator should operate in classification mode.
        During classification, the targets are encoded.

    encoder: Optional[BaseEstimator]
        Host a encoder object if the data requires transformation (for example, if
        provided a categorical column in a pandas DataFrame).

    enc_columns: List[str]
        List of columns that where encoded
    """

    def __init__(
        self,
        is_classification: bool = False,
        logger: Optional[PickableLoggerAdapter] = None,
    ) -> None:
        self.is_classification = is_classification

        self.data_type = None  # type: Optional[type]

        # NOTE: type update
        #   Encoders don't seems to have a unified base class for
        #   the methods 'transform'. Could make a `prototype` class that
        #   duck types for the 'transform', 'fit_transform' methods
        #   This is however fine while we only use ordinal encoder
        self.encoder = None  # type: Optional[OrdinalEncoder]

        self.out_dimensionality = None  # type: Optional[int]
        self.type_of_target = None  # type: Optional[str]

        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Store the dtype for remapping to correct type
        self.dtype = None  # type: Optional[Type]

        self._is_fitted = False

    def fit(
        self,
        y_train: Union[List, np.ndarray, pd.Series, pd.DataFrame],
        y_test: Optional[Union[List, np.ndarray, pd.Series, pd.DataFrame]] = None,
    ) -> "TargetValidator":
        """
        Validates and fit a categorical encoder (if needed) to the targets
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            y_train: SUPPORTED_TARGET_TYPES
                A set of targets set aside for training
            y_test: Union[SUPPORTED_TARGET_TYPES]
                A hold out set of data used of the targets. It is also used to fit the
                categories of the encoder.
        """
        # Check that the data is valid
        self._check_data(y_train)

        shape = np.shape(y_train)
        if y_test is not None:
            self._check_data(y_test)

            if len(shape) != len(np.shape(y_test)) or (
                len(shape) > 1 and (shape[1] != np.shape(y_test)[1])
            ):
                raise ValueError(
                    "The dimensionality of the train and test"
                    " targets do not match"
                    f" train {np.shape(y_train)}"
                    f" != test {np.shape(y_test)}"
                )

            if isinstance(y_train, pd.DataFrame):
                if not isinstance(y_test, pd.DataFrame):
                    y_test = pd.DataFrame(y_test)

                if y_train.columns.tolist() != y_test.columns.tolist():
                    raise ValueError(
                        "Train and test targets must both have the"
                        f" same columns, yet y={y_train.columns}"
                        f" and y_test={y_test.columns}"
                    )

                if list(y_train.dtypes) != list(y_test.dtypes):
                    raise ValueError(
                        "Train and test targets must both have the same dtypes"
                    )

        if self.out_dimensionality is None:
            self.out_dimensionality = 1 if len(shape) == 1 else shape[1]
        else:
            _n_outputs = 1 if len(shape) == 1 else shape[1]
            if self.out_dimensionality != _n_outputs:
                raise ValueError(
                    "Number of outputs changed from %d to %d!"
                    % (self.out_dimensionality, _n_outputs)
                )

        # Fit on the training data
        self._fit(y_train, y_test)

        self._is_fitted = True

        return self

    def _fit(
        self,
        y_train: Union[List, np.ndarray, pd.Series, pd.DataFrame],
        y_test: Optional[Union[List, np.ndarray, pd.Series, pd.DataFrame]] = None,
    ) -> "TargetValidator":
        """
        If dealing with classification, this utility encodes the targets.

        It does so by also using the classes from the test data, to prevent encoding
        errors

        Parameters
        ----------
            y_train: SUPPORTED_TARGET_TYPES
                The labels of the current task. They are going to be encoded in case
                of classification
            y_test: Optional[SUPPORTED_TARGET_TYPES]
                A holdout set of labels
        """
        if not self.is_classification or self.type_of_target == "multilabel-indicator":
            # Only fit an encoder for classification tasks
            # Also, encoding multilabel indicator data makes the data multiclass
            # Let the user employ a MultiLabelBinarizer if needed
            self.encoder = None
            return self

        # Validate the shape of the input
        shape = np.shape(y_train)
        ndim = len(shape)
        if ndim > 1 and shape[1] != 1:
            # We should not reach this if statement, we check for type of targets before
            raise ValueError(
                "Multi-dimensional classification is not yet"
                " supported. Encoding multidimensional data"
                " converts multiple columns to a 1 dimensional encoding."
                f" Data involved = {shape}/{self.type_of_target}"
            )

        # Create the encoder
        self.encoder = OrdinalEncoder(handle_unknown="error")

        # Clear typing to just numpy arrays and pandas
        if isinstance(y_train, List):
            y = np.asarray(y_train)
        else:
            y = y_train

        # Attempt to store dtype if it's numeric, we use this in
        # inverse_transform to try corretly restore it's dtype
        if isinstance(y, pd.Series):
            if isinstance(y.dtype, ExtensionDtype):
                warnings.warn(
                    "Fitting transformer with a pandas series which"
                    f" has the dtype {y.dtype}. Inverse transform"
                    " may not be able preserve dtype when converting"
                    " to np.ndarray"
                )
            if is_numeric_dtype(y.dtype):
                self.dtype = y.dtype
        elif isinstance(y, pd.DataFrame):
            if is_numeric_dtype(y.dtypes[0]):
                self.dtype = y.dtypes[0]
        else:
            if is_numeric_dtype(y.dtype):
                self.dtype = y.dtype

        # Merge y_test and y_train for encoding
        if y_test is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                if isinstance(y, pd.Series):
                    y_test = pd.Series(y_test)
                else:
                    y_test = pd.DataFrame(y_test)

                y = pd.concat([y, y_test], ignore_index=True, sort=False)
            else:
                y = np.concatenate([y, y_test])

        # If one dimensional, we need to reshape it
        # [1, 2, 3] -> [[1], [2], [3]]
        if y.ndim == 1:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()

            y = y.reshape(-1, 1)

        # Fit encoder
        self.encoder.fit(y)

        return self

    def transform(
        self,
        y: Union[List, pd.Series, pd.DataFrame, np.ndarray],
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
            raise NotFittedError("TargetValidator must have fit() called first")

        # Check the data here so we catch problems on new test data
        self._check_data(y)

        # Clear the types List and DataFrame off of y
        if isinstance(y, List):
            y_transformed = np.asarray(y)
        elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y_transformed = y.to_numpy()
        else:
            y_transformed = y

        if self.encoder is None:
            return y_transformed

        # The Ordinal encoder expects a 2 dimensional input.
        # The targets are 1 dimensional, so reshape to match the expected shape
        if y_transformed.ndim == 1:
            y_transformed = y_transformed.reshape(-1, 1)
            y_transformed = self.encoder.transform(y_transformed).reshape(-1)
        else:
            y_transformed = self.encoder.transform(y_transformed)

        # check_array ensures correct numerical features for array
        y_transformed = sklearn.utils.check_array(
            y_transformed,
            force_all_finite=True,
            accept_sparse="csr",
            ensure_2d=False,
        )

        # Ensure we flatten any arrays [[1], [2], [1], [1], ...] to [1,2,1,1,...]
        if y_transformed.ndim == 2 and y_transformed.shape[1] == 1:
            y_transformed = y_transformed.ravel()

        return y_transformed

    def inverse_transform(
        self,
        y: Union[List, pd.Series, pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Revert any encoding transformation done on a target array.

        Parameters
        ----------
        y: Union[List, np.ndarray, pd.DataFrame, pd.Series]
            Target array to be transformed back to original form before encoding

        Return
        ------
        np.ndarray:
            The transformed array
        """
        if not self._is_fitted:
            raise NotFittedError("TargetValidator must have fit() called first")

        # If there's no encoder, just return it as the expected ndarray
        if self.encoder is None:
            return np.asarray(y)

        # If it's one dimensional, we need to convert it
        # [1,2,3] -> [[1], [2], [3]]
        shape = np.shape(y)
        if len(shape) == 1:

            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()
            elif isinstance(y, List):
                y = np.asarray(y)

            # Perform the transformation and reshape it
            y = y.reshape(-1, 1)
            y_inv = self.encoder.inverse_transform(y)
            y_inv = y_inv.reshape(-1)

        else:
            # Otherwise we can can just invert it as expected
            y_inv = self.encoder.inverse_transform(y)

        # Inverse transform returns a numpy array of type object
        # This breaks certain metrics as accuracy, which makes type_of_target be unknown
        # If while fit a dtype was observed, we try to honor that dtype
        if self.dtype is not None:
            y_inv = y_inv.astype(self.dtype)

        return y_inv

    def is_single_column_target(self) -> bool:
        """Output is encoded with a single column encoding"""
        return self.out_dimensionality == 1

    def _check_data(
        self,
        y: SUPPORTED_TARGET_TYPES,
    ) -> None:
        """
        Perform dimensionality and data type checks on the targets

        Parameters
        ----------
        y: Union[np.ndarray, pd.DataFrame, pd.Series]
            A set of features whose dimensionality and data type is going to be checked
        """
        if not isinstance(
            y, (np.ndarray, pd.DataFrame, List, pd.Series)
        ) and not isinstance(y, spmatrix):
            raise ValueError(
                "Auto-sklearn only supports Numpy arrays, Pandas DataFrames,"
                " pd.Series, sparse data and Python Lists as targets, yet, "
                "the provided input is of type {}".format(type(y))
            )

        if isinstance(y, spmatrix) and not np.issubdtype(y.dtype.type, np.number):
            raise ValueError(
                "When providing a sparse matrix as targets, the only supported "
                "values are numerical. Please consider using a dense"
                " instead."
            )

        if self.data_type is None:
            self.data_type = type(y)
        if self.data_type != type(y):
            self.logger.warning(
                "Auto-sklearn previously received targets of type %s "
                "yet the current features have type %s. Changing the dtype "
                "of inputs to an estimator might cause problems"
                % (
                    str(self.data_type),
                    str(type(y)),
                ),
            )

        # No Nan is supported
        has_nan_values = False
        if hasattr(y, "iloc"):
            has_nan_values = cast(pd.DataFrame, y).isnull().values.any()

        if isinstance(y, spmatrix):
            has_nan_values = not np.array_equal(y.data, y.data)
        else:
            # List and array like values are considered here
            # np.isnan cannot work on strings, so we have to check for every element
            # but NaN, are not equal to themselves:
            has_nan_values = not np.array_equal(y, y)
        if has_nan_values:
            raise ValueError(
                "Target values cannot contain missing/NaN values. "
                "This is not supported by scikit-learn. "
            )

        # Pandas Series is not supported for multi-label indicator
        # This format checks are done by type of target
        try:
            self.type_of_target = type_of_target(y)
        except Exception as e:
            raise ValueError(
                "The provided data could not be interpreted by Sklearn. "
                "While determining the type of the targets via type_of_target "
                "run into exception: {}.".format(e)
            )

        supported_output_types = (
            "binary",
            "continuous",
            "continuous-multioutput",
            "multiclass",
            "multilabel-indicator",
            # Notice unknown/multiclass-multioutput are not supported
            # This can only happen during testing only as estimators
            # should filter out unsupported types.
        )
        if self.type_of_target not in supported_output_types:
            raise ValueError(
                "Provided targets are not supported by Auto-Sklearn. "
                "Provided type is {} whereas supported types are {}.".format(
                    self.type_of_target, supported_output_types
                )
            )

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
