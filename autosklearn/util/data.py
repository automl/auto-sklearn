import warnings
from collections.abc import Mapping
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

from sklearn.model_selection import train_test_split

from autosklearn.data.validation import SUPPORTED_FEAT_TYPES
from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit


class _DtypeReductionMapping(Mapping):
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
    _mapping: Dict[type, type] = {
        np.float32: np.float32,
        np.float64: np.float32,
    }

    # In spite of the names, np.float96 and np.float128
    # provide only as much precision as np.longdouble,
    # that is, 80 bits on most x86 machines and 64 bits
    # in standard Windows builds.
    if hasattr(np, 'float96'):
        _mapping[np.float96] = np.float64

    if hasattr(np, 'float128'):
        _mapping[np.float128] = np.float64

    @classmethod
    def __getitem__(cls, item: type) -> type:
        for k, v in cls._mapping.items():
            if k == item:
                return v
        raise KeyError(item)

    @classmethod
    def __iter__(cls) -> Iterator[type]:
        return iter(cls._mapping.keys())

    @classmethod
    def __len__(cls) -> int:
        return len(cls._mapping)


reduction_mapping = _DtypeReductionMapping()
supported_precision_reductions = list(reduction_mapping)


def binarization(array: Union[List, np.ndarray]) -> np.ndarray:
    # Takes a binary-class datafile and turn the max value (positive class)
    # into 1 and the min into 0
    array = np.array(array, dtype=float)  # conversion needed to use np.inf
    if len(np.unique(array)) > 2:
        raise ValueError('The argument must be a binary-class datafile. '
                         '{} classes detected'.format(len(np.unique(array))))

    # manipulation which aims at avoid error in data
    # with for example classes '1' and '2'.
    array[array == np.amax(array)] = np.inf
    array[array == np.amin(array)] = 0
    array[array == np.inf] = 1
    return np.array(array, dtype=int)


def multilabel_to_multiclass(array: Union[List, np.ndarray]) -> np.ndarray:
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


def convert_to_bin(Ycont: List, nval: int, verbose: bool = True) -> List:
    # Convert numeric vector to binary (typically classification target values)
    if verbose:
        pass
    Ybin = [[0] * nval for _ in range(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[np.int(Ycont[i])] = 1
        Ybin[i] = line
    return Ybin


def predict_RAM_usage(X: np.ndarray, categorical: List[bool]) -> float:
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


def subsample(
    X: Union[np.ndarray, spmatrix],
    y: np.ndarray,
    is_classification: bool,
    sample_size: Union[float, int],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[SUPPORTED_FEAT_TYPES, Union[List, np.ndarray, pd.DataFrame, pd.Series]]:
    """ Subsamples data returning the same type as it recieved.

    If `is_classification`, we split using a stratified shuffle split which
    preserves unique labels in the training set.

    NOTE:
    It's highly unadvisable to use lists here. In order to preserver types,
    we convert to a numpy array and then back to a list.

    NOTE2:
    Interestingly enough, StratifiedShuffleSplut and descendants don't support
    sparse `y` in `split(): _check_array` call. Hence, neither do we.

    Parameters
    ----------
    X: Union[np.ndarray, spmatrix]
        The X's to subsample

    y: np.ndarray
        The Y's to subsample

    is_classification: bool
        Whether this is classification data or regression data. Required for
        knowing how to split.

    sample_size: float | int
        If float, percentage of data to take otherwise if int, an absolute
        count of samples to take.

    random_state: int | RandomState = None
        The random state to pass to the splitted

    Returns
    -------
    (np.ndarray | spmatrix, np.ndarray)
        The X and y subsampled according to sample_size
    """
    if isinstance(X, List):
        X = np.asarray(X)

    if isinstance(y, List):
        y = np.asarray(y)

    if is_classification:
        splitter = CustomStratifiedShuffleSplit(
            train_size=sample_size,
            random_state=random_state
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
        X, _, y, _ = train_test_split(  # type: ignore
            X, y,
            train_size=sample_size,
            random_state=random_state,
        )

    return X, y


def reduce_precision(
    X: Union[np.ndarray, spmatrix]
) -> Tuple[Union[np.ndarray, spmatrix], Type]:
    """ Reduces the precision of a np.ndarray or spmatrix containing floats

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
        raise ValueError(f"X.dtype = {X.dtype} not equal to any supported"
                         f" {supported_precision_reductions}")

    precision = reduction_mapping[X.dtype]
    return X.astype(precision), precision


def reduce_dataset_size_if_too_large(
    X: Union[np.ndarray, spmatrix],
    y: np.ndarray,
    memory_limit: int,
    is_classification: bool,
    random_state: Union[int, np.random.RandomState] = None,
    operations: List[str] = ['precision', 'subsample'],
    multiplier: Union[float, int] = 8,
) -> Tuple[Union[np.ndarray, spmatrix], np.ndarray]:
    f""" Reduces the size of the dataset if it's too close to the memory limit.

    Follows the order of the operations passed in and retains the type of its
    input.

    Precision reduction will only work on the following float types:
    -   {supported_precision_reductions}

    Subsampling will ensure that the memory limit is satisfied while precision
    reduction will only perform one level of precision reduction. Technically,
    you could supply multiple rounds of precision reduction, i.e. to reduce
    np.float128 to np.float32 you could use `operations = ['precision'] * 2`.
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

    memory_limit: int
        The amount of memory allocated in megabytes

    is_classification: bool
        Whether it's a classificaiton dataset or not. This is important when
        considering how to subsample.

    random_state: int | RandomState = None
        The random_state to use for subsampling.

    operations: List[str] = ['precision', 'subsampling']
        A list of operations that are permitted to be performed to reduce
        the size of the dataset.

    multiplier: float | int = 8
        When performing reductions, satisfies the conditions that:
        * Reduce precision if `size(X) * multiplier >= memory_limit`
        * Subsample so that `size(X) * mulitplier = memory_limit` is satisfied``

    Returns
    -------
    (spmatrix | np.ndarray, np.ndarray)
        The reduced X, y if reductions were needed
    """
    # Validation
    assert memory_limit > 0

    if 'precision' in operations and X.dtype not in supported_precision_reductions:
        raise ValueError(f"Unsupported type `{X.dtype}` for precision reduction")

    def megabytes(arr: Union[np.ndarray, spmatrix]) -> float:
        return 1e-6 * (arr.nbytes if isinstance(X, np.ndarray) else arr.data.nbytes)

    for operation in operations:

        if operation == 'precision':
            # If `multiplier` times the dataset is too big for memory, we try
            # to reduce the precision if it's a high precision dataset
            if megabytes(X) * multiplier > memory_limit:
                X, precision = reduce_precision(X)
                warnings.warn(
                    f'Dataset too large for memory limit {memory_limit}MB, '
                    f'reduced the precision from {X.dtype} to {precision}',
                )

        elif operation == 'subsample':
            # If the dataset is still too big such that we couldn't fit
            # `multiplier` of them in memory, we subsample such that we can
            if memory_limit < megabytes(X) * multiplier:

                n_samples_before = X.shape[0]
                reduction_percent = float(memory_limit) / (megabytes(X) * multiplier)

                # NOTE: type ignore
                #
                # Tried the generic `def subsample(X: T) -> T` approach but it was
                # failing elsewhere, keeping it simple for now
                X, y = subsample(  # type: ignore
                    X, y,
                    sample_size=reduction_percent,
                    is_classification=is_classification,
                    random_state=random_state
                )

                n_samples_after = X.shape[0]
                warnings.warn(
                    f"Dataset too large for memory limit {memory_limit}MB, reduced"
                    f" number of samples from {n_samples_before} to {n_samples_after}."
                )

        else:
            raise ValueError(f"Unknown operation `{operation}`")
    return X, y
