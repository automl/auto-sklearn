import typing
import warnings

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

import scipy.sparse

import sklearn.utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer


class TargetValidator(BaseEstimator):

    def __init__(self) -> None:
        # If a dataframe was provided, we populate
        # this attribute with the column types from the dataframe
        # That is, this attribute contains whether autosklearn
        # should treat a column as categorical or numerical
        # During fit, if the user provided feat_types, the user
        # constrain is honored. If not, this attribute is used.
        self.feature_types = None  # type: typing.Optional[typing.List[str]]

        self.data_type = None  # type: typing.Optional[type]

        self.encoder = None  # type: typing.Optional[BaseEstimator]
        self.enc_columns = []  # type: typing.List[str]

        self.output_dimensionality = None  # type: Optional[int]

        self._is_fitted = False

    def fit(
        self,
        y_train: typing.Union[np.ndarray, pd.DataFrame],
        y_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
        is_classification: bool = False,
    ) -> BaseEstimator:
        """
        Validates and fit a categorical encoder (if needed) to the targets
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            y_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of targets set aside for training
            y_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of data used of the targets. It is also used to fit the
                categories of the encoder.
            is_classification: bool
                A bool that indicates if the validator should operate in classification mode.
                During classification, the targets are encoded.
        """
        # Check that the data is valid
        self.check_data(y_train, y_test)

        # Fit on the training data
        self._fit(y_train, y_test, is_classification)

        self._is_fitted = True

        return self

    def _fit(
        self,
        y_train: typing.Union[np.ndarray, pd.DataFrame],
        y_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]] = None,
        is_classification: bool = False,
    ) -> BaseEstimator:
        """
        If dealing with classification, this utility encodes the targets.

        It does so by also using the classes from the test data, to prevent enconding
        errors

        Parameters
        ----------
            y_train: typing.Union[np.ndarray, pd.DataFrame]
                The labels of the current task. They are going to be encoded in case
                of classification
            y_test: typing.Union[np.ndarray, pd.DataFrame]
                A holdout set of labels
            is_classification: bool
                A bool that indicates if the validator should operate in classification mode.
                During classification, the targets are encoded.
        """

        if not is_classification:
            # Only fit an encoder for classification tasks
            return self

        if y_test is not None:
            if hasattr(y_train, "iloc"):
                y_train = pd.concat([y_train, y_test], ignore_index=True, sort=False)
            elif isinstance(y_train, list):
                y_train = y_train + y_test
            elif isinstance(y_train, np.ndarray):
                y_train = np.concatenate((y_train, y_test))

        if y_train.ndim == 1 or (y_train.ndim > 1 and y_train.shape[1] == 1):
            # The label encoder makes sure data is, and remains
            # 1 dimensional
            self.encoder = preprocessing.LabelEncoder()
        else:
            self.encoder = make_column_transformer(
                (preprocessing.OrdinalEncoder(), list(range(y_train.shape[1]))),
            )

        # Mypy redefinition
        assert self.encoder is not None

        # remove ravel warning from pandas Series
        if len(y_train.shape) > 1 and y_train.shape[1] == 1 and hasattr(y_train, "to_numpy"):
            self.encoder.fit(y_train.to_numpy().ravel())
        else:
            self.encoder.fit(y_train)

        return self

    def transform(
        self,
        y: typing.Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Validates and fit a categorical encoder (if needed) to the features.
        The supported data types are List, numpy arrays and pandas DataFrames.

        Parameters
        ----------
            y: typing.Union[np.ndarray, pd.DataFrame]
                A set of targets that are going to be encoded if the current task
                is classification
        Return
        ------
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise ValueError("Cannot call transform on a validator that is not fitted")

        if self.encoder is not None:
            try:
                # remove ravel warning from pandas Series
                if len(y.shape) > 1 and y.shape[1] == 1 and hasattr(y, "to_numpy"):
                    y = self.encoder.transform(y.to_numpy().ravel())
                else:
                    y = self.encoder.transform(y)
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
                            self.encoder.transformers_[0][1].categories_,
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
                            self.encoder.classes_,
                            e.args[0],
                        )
                    )
                else:
                    raise e

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
        y: typing.Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Revert any encoding transformation done on a target array

        Parameters
        ----------
            y: typing.Union[np.ndarray, pd.DataFrame]
                Target array to be transformed back to original form before encoding
        Return
        ------
            np.ndarray:
                The transformed array
        """
        if not self._is_fitted:
            raise ValueError("Cannot call inverse_transform on a validator that is not fitted")

        if self.encoder is None:
            return y

        # Handle different ndim encoder for target
        if hasattr(self.encoder, 'inverse_transform'):
            return self.encoder.inverse_transform(y)
        else:
            return self.encoder.named_transformers_['ordinalencoder'].inverse_transform(y)

    def check_data(
        self,
        y_train: typing.Union[np.ndarray, pd.DataFrame],
        y_test: typing.Optional[typing.Union[np.ndarray, pd.DataFrame]],
    ) -> None:
        """
        Makes sure the targets comply with auto-sklearn data requirements and
        checks y_train/y_test dimensionality

        This method also stores metadata for future checks

        Parameters
        ----------
            y_train: typing.Union[np.ndarray, pd.DataFrame]
                A set of targets set aside for training
            y_test: typing.Union[np.ndarray, pd.DataFrame]
                A hold out set of data used for checking
        """

        self._check_data(y_train)

        if y_test is not None:
            self._check_data(y_test)

            if len(np.shape(y_train)) != len(np.shape(y_test)):
                raise ValueError("The dimensionality of the train and test targets "
                                 "does not match train({}) != test({})".format(
                                     np.shape(y_train),
                                     np.shape(y_test)
                                 ))
            if isinstance(y_train, pd.DataFrame):
                if y_train.columns != y_test.columns:
                    raise ValueError(
                        "Train and test targets must both have the same columns, yet "
                        "y={} and y_test={} ".format(
                            y_train.columns,
                            y_test.columns
                        )
                    )

                if list(y_train.dtypes) != list(y_test.dtypes):
                    raise ValueError("Train and test targets must both have the same dtypes")

        if self.output_dimensionality is None:
            self.output_dimensionality = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        else:
            _n_outputs = 1 if len(y_train.shape) == 1 else y_train.shape[1]
            if self.output_dimensionality != _n_outputs:
                raise ValueError('Number of outputs changed from %d to %d!' %
                                 (self.output_dimensionality, _n_outputs))

    def is_single_column_target(self) -> bool:
        """
        Output is encoded with a single column encoding
        """
        return self.output_dimensionality == 1

    def _check_data(
        self,
        y: typing.Union[np.ndarray, pd.DataFrame],
    ) -> None:
        """
        Perform dimensionality and data type checks on the targets

        Parameters
        ----------
            y: typing.Union[np.ndarray, pd.DataFrame]
                A set of features whose dimensionality and data type is going to be checked
        """

        if not isinstance(y, (np.ndarray, pd.DataFrame, list, pd.Series)):
            raise ValueError("Auto-sklearn only supports Numpy arrays, Pandas DataFrames,"
                             " pd.Series and Python Lists as targets, yet, the provided input is"
                             " of type {}".format(
                                 type(y)
                             ))

        if self.data_type is None:
            self.data_type = type(y)
        if self.data_type != type(y):
            warnings.warn("Auto-sklearn previously received features of type %s "
                          "yet the current features have type %s. Changing the dtype "
                          "of inputs to an estimator might cause problems" % (
                                str(self.data_type),
                                str(type(y)),
                             ),
                          )

        # Do not support category/string numpy data. Only numbers
        if hasattr(y, "dtype"):
            if not np.issubdtype(y.dtype.type, np.number):  # type: ignore[union-attr]
                raise ValueError(
                    "When providing a numpy array as targets Auto-sklearn, the only valid "
                    "dtypes are numerical ones. The provided data type {} is not supported."
                    "".format(
                        y.dtype.type,  # type: ignore[union-attr]
                    )
                )

        # No Nan is supported
        if np.any(pd.isnull(y)):
            raise ValueError("Target values cannot contain missing/NaN values. "
                             "This is not supported by scikit-learn. "
                             )

        # Target data as sparse is not supported
        if scipy.sparse.issparse(y):
            raise ValueError("Unsupported target data provided"
                             "Input targets to auto-sklearn must not be of "
                             "type sparse. Please convert the target input (y) "
                             "to a dense array via scipy.sparse.csr_matrix.todense(). "
                             )
