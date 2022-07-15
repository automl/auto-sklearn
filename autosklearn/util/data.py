from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, TypeVar

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.model_selection import train_test_split

from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit

T = TypeVar("T", np.ndarray, spmatrix)


def megabytes(arr: np.ndarray | spmatrix) -> float:
    """Get the megabyte usage of some data"""
    return (arr.nbytes if isinstance(arr, np.ndarray) else arr.data.nbytes) / (2**20)


class _DtypeReductionMapping(Mapping[type, type]):
    """
    Unfortuantly, mappings compare by hash(item) and not the __eq__ operator
    between the key and the item.

    Hence we wrap the dict in a Mapping class and implement our own __getitem__
    such that we do use __eq__ between keys and query items.

    >>> np.float32 == dtype('float32') # True, they are considered equal
    >>>
    >>> mydict = { np.float32: 'hello' }
    >>>
    >>> # Equal by __eq__ but dict operations fail
    >>> np.dtype('float32') in mydict # False
    >>> mydict[dtype('float32')]  # KeyError

    This mapping class fixes that supporting the `in` operator as well as `__getitem__`

    >>> reduction_mapping = _DtypeReductionMapping()
    >>>
    >>> reduction_mapping[np.dtype('float64')] # np.float32
    >>> np.dtype('float32') in reduction_mapping # True
    """

    # Information about dtype support
    _mapping: dict[type, type] = {
        np.float32: np.float32,
        np.float64: np.float32,
    }

    # In spite of the names, np.float96 and np.float128 provide only as much precision
    # as np.longdouble, that is, 80 bits on most x86 machines and 64 bits in Windows.
    if hasattr(np, "float96"):
        _mapping[np.float96] = np.float64

    if hasattr(np, "float128"):
        _mapping[np.float128] = np.float64

    @classmethod
    def __getitem__(cls, item: type) -> type:
        v = next((v for k, v in cls._mapping.items() if k == item), None)
        if v is None:
            raise KeyError(item)
        return v

    @classmethod
    def __iter__(cls) -> Iterator[type]:
        return iter(cls._mapping.keys())

    @classmethod
    def __len__(cls) -> int:
        return len(cls._mapping)


reduction_mapping = _DtypeReductionMapping()
supported_precision_reductions = tuple(reduction_mapping)


class DatasetCompression:

    _valid_methods = ("precision", "subsample")
    supported_precision_reductions = supported_precision_reductions

    def __init__(
        self,
        *,
        limit: int,
        allocation: int | float = 0.1,
        methods: Iterable[str] = ("precisions", "subsample"),
    ):
        self.limit = limit
        self.allocation = allocation
        self.methods = methods

        if not any(methods):
            raise ValueError("No methods passed for dataset compression")

        invalid_methods = [m for m in self.methods if m not in self._valid_methods]
        if any(invalid_methods):
            raise ValueError(
                f"Unrecognized `methods` {invalid_methods},"
                f"must be in {self._valid_methods}"
            )

        mem = self.allocation
        if isinstance(mem, float):
            if 0 < mem < 1:
                self.mem_usage = allocation * mem
            else:
                raise ValueError(f"`memory_allocation` float ({mem}) must be in (0, 1)")
        else:
            if 0 < mem < limit:
                self.mem_usage = allocation
            else:
                raise ValueError(f"`memory_allocation` int ({mem}) must be < {limit}")

    @staticmethod
    def subsample(
        X: T,
        y: np.ndarray,
        *,
        size: float | int,
        stratify: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> tuple[T, np.ndarray]:
        """Subsamples data returning the same type as it recieved.

        NOTE:
        It's highly unadvisable to use lists here. In order to preserver types,
        we convert to a numpy array and then back to a list.

        NOTE2:
        Interestingly enough, StratifiedShuffleSplut and descendants don't support
        sparse `y` in `split(): _check_array` call. Hence, neither do we.

        Parameters
        ----------
        X: np.ndarray | spmatrix
            The X's to subsample

        y: np.ndarray
            The Y's to subsample

        stratify: bool = False
            Whether this is classification data or regression data.
            Required for knowing how to split.

        sample_size: float | int
            If float, percentage of data to take otherwise if int, an absolute
            count of samples to take.

        random_state: int | RandomState | None = None
            The random state to pass to the splitted

        Returns
        -------
        (np.ndarray | spmatrix, np.ndarray)
            The X and y subsampled according to sample_size
        """
        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(y, list):
            y = np.asarray(y)

        if stratify:
            splitter = CustomStratifiedShuffleSplit(
                train_size=size, random_state=random_state
            )
            left_idxs, _ = next(splitter.split(X=X, y=y))

            # This function supports pandas objects but they won't get here
            # yet as we do not reduce the size of pandas dataframes.
            if isinstance(X, pd.DataFrame):
                idxs = X.index[left_idxs]
                X = X.loc[idxs]
            else:
                X = X[left_idxs]

            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                idxs = y.index[left_idxs]
                y = y.loc[idxs]
            else:
                y = y[left_idxs]

        else:
            X, _, y, _ = train_test_split(
                X, y, train_size=size, random_state=random_state
            )

        return X, y

    @classmethod
    def reduce_precision(cls, X: T) -> tuple[T, type]:
        """Reduces the precision of a np.ndarray or spmatrix containing floats

        Parameters
        ----------
        X:  np.ndarray | spmatrix
            The data to reduce precision of.

        Returns
        -------
        (ndarray | spmatrix, dtype)
            Returns the reduced data X along with the dtype it was reduced to.
        """
        if X.dtype not in supported_precision_reductions:
            raise ValueError(
                f"X.dtype = {X.dtype} not equal to any supported"
                f" {supported_precision_reductions}"
            )

        precision = reduction_mapping[X.dtype]
        return X.astype(precision), precision

    def compress(
        self,
        X: T,
        y: np.ndarray,
        *,
        stratify: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> tuple[T, np.ndarray]:
        """Reduces the size of the dataset if it's too close to the memory limit.

        Follows the order of the method passed in and retains the type of its
        input.

        Subsampling will ensure that the memory limit is satisfied while precision
        reduction will only perform one level of precision reduction.
        Technically, you could supply multiple rounds of precision reduction,

        However, if that's the use case, it'd be advised to simply use the function
        `autosklearn.util.data.reduce_precision`.

        NOTE: limitations
        * Does not support dataframes yet
            -   Requires implementing column wise precision reduction
            -   Requires calculating memory usage

        Parameters
        ----------
        X: np.ndarray | spmatrix
            The features of the dataset.

        y: np.ndarray
            The labels of the dataset.

        stratify: bool = False
            Whether to stratify if subsampling

        random_state: int | RandomState | None = None
            The random_state to use for subsampling.

        Returns
        -------
        (spmatrix | np.ndarray, np.ndarray)
            The reduced X, y if reductions were needed
        """
        for method in self.methods:

            if method == "precision":

                if X.dtype not in self.supported_precision_reductions:
                    warnings.warn(f"`precision` method on {X.dtype} not supported")
                    continue

                # If the dataset is too big for the allocated memory,
                # we then try to reduce the precision if it's a high precision dataset
                if megabytes(X) > self.mem_usage:
                    X, precision = self.reduce_precision(X)
                    warnings.warn(
                        f"Dataset too large for allocated memory {self.mem_usage}MB, "
                        f"reduced the precision from {X.dtype} to {precision}",
                    )

            elif method == "subsample":
                # If the dataset is still too big such that we couldn't fit
                # into the allocated memory, we subsample it so that it does
                if megabytes(X) > self.mem_usage:

                    n_samples_before = X.shape[0]
                    sample_percentage = self.mem_usage / megabytes(X)

                    X, y = self.subsample(
                        X,
                        y,
                        size=sample_percentage,
                        stratify=stratify,
                        random_state=random_state,
                    )

                    n_samples_after = X.shape[0]
                    warnings.warn(
                        f"Dataset too large for allocated memory {self.mem_usage}MB,"
                        f" reduced number of samples from {n_samples_before} to"
                        f" {n_samples_after}."
                    )

            else:
                raise ValueError(f"Unknown method `{method}`")

        return X, y

    @classmethod
    def supports(self, X: Any, y: Any) -> bool:
        # Currently don't support pd.Series or pd.Dataframe
        return isinstance(X, (np.ndarray, spmatrix)) and isinstance(y, np.ndarray)


def binarization(array: list | np.ndarray) -> np.ndarray:
    # Takes a binary-class datafile and turn the max value (positive class)
    # into 1 and the min into 0
    array = np.array(array, dtype=float)  # conversion needed to use np.inf
    if len(np.unique(array)) > 2:
        raise ValueError(
            "The argument must be a binary-class datafile. "
            "{} classes detected".format(len(np.unique(array)))
        )

    # manipulation which aims at avoid error in data
    # with for example classes '1' and '2'.
    array[array == np.amax(array)] = np.inf
    array[array == np.amin(array)] = 0
    array[array == np.inf] = 1
    return np.array(array, dtype=int)


def multilabel_to_multiclass(array: list | np.ndarray) -> np.ndarray:
    array = binarization(array)
    return np.array([np.nonzero(array[i, :])[0][0] for i in range(len(array))])


def convert_to_num(Ybin: np.ndarray) -> np.ndarray:
    """
    Convert binary targets to numeric vector
    typically classification target values
    :param Ybin:
    :return:
    """
    result = np.array(Ybin)
    if len(Ybin.shape) != 1:
        result = np.dot(Ybin, range(Ybin.shape[1]))
    return result


def convert_to_bin(Ycont: list, nval: int, verbose: bool = True) -> list:
    # Convert numeric vector to binary (typically classification target values)
    if verbose:
        pass
    Ybin = [[0] * nval for _ in range(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[int(Ycont[i])] = 1
        Ybin[i] = line
    return Ybin


def predict_RAM_usage(X: np.ndarray, categorical: list[bool]) -> float:
    # Return estimated RAM usage of dataset after OneHotEncoding in bytes.
    estimated_columns = 0
    for i, cat in enumerate(categorical):
        if cat:
            unique_values = np.unique(X[:, i])
            num_unique_values = np.sum(np.isfinite(unique_values))
            estimated_columns += num_unique_values
        else:
            estimated_columns += 1
    estimated_ram = estimated_columns * X.shape[0] * X.dtype.itemsize
    return estimated_ram
