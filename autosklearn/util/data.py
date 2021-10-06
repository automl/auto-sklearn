import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

import pandas as pd

from scipy.sparse import spmatrix

from sklearn.model_selection import train_test_split

from autosklearn.data.validation import SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES
from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit

supported_dtypes = [np.float32, np.float64]
supported_dtypes.append(np.float96 if hasattr(np, 'float96') else None)
supported_dtypes.append(np.float128 if hasattr(np, 'float128') else None)

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
    Ybin = [[0] * nval for x in range(len(Ycont))]
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

def byte_size(dtype: np.dtype) -> int:
    """ Returns the amount of bytes required for an element of dtype

    Only supports np.float{32,64,96,128} for now.

    Parameters
    ----------
    dtype: np.dtype
        the dtype to estimate bits for

    safe: bool = True
        Whether to raise an error if the dtype is unknown

    Returns
    -------
    int
        The number of bytes
    """
    if dtype not in supported_dtypes:
        raise ValueError(f"{dtype} not in supported {supported_dtypes}")

    mapping: Dict[Any, int] = {
        np.float32: 4,
        np.float64: 8,
    }
    # In spite of the names, np.float96 and np.float128
    # provide only as much precision as np.longdouble,
    # that is, 80 bits on most x86 machines and 64 bits
    # in standard Windows builds.
    if hasattr(np, 'float96'):
        mapping[np.float96] = 16

    if hasattr(np, 'float128'):
        mapping[np.float128] = 16

    return mapping[dtype]

def megabytes(
    X: Union[spmatrix, np.ndarray]
) -> float:
    """ Estimate how large X is in megabytes

    If ndarray is not a uniform type, this estimation does not work.
    Also does not support pandas dataframes for now as a result

    Parameters
    ----------
    X: ndarray | spmatrix
        The data to estimate the size of
    """
    if X.dtype not in supported_dtypes:
        raise ValueError(f"{X.dtype} not in {supported_dtypes}")

    return X.shape[0] * X.shape[1] * byte_size(X.dtype) * 1e-6

XT = TypeVar("XT", bound=SUPPORTED_FEAT_TYPES)
YT = TypeVar("YT", bound=SUPPORTED_TARGET_TYPES)
def subsample(
    X: XT,
    y: YT,
    is_classification: bool,
    sample_size: Union[float, int],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> Tuple[XT, YT]:
    """ Subsamples the array so it fits into the memory limit

    Returns the same type as it recieved where
    *   XT is bound to [List, np.ndarray, spmatrix, pd.DataFrame]
    *   YT is bound to [List, np.ndarray, spmatrix, pd.Series, pd.DataFrame]

    Parameters
    ----------
    X: XT
        The X's to subsample

    Y: YT
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
    (XT, YT)
        The X and y subsampled
    """
    if is_classification:
        splitter = CustomStratifiedShuffleSplit(
            train_size=sample_size,
            random_state=random_state
        )
        left_idxs, _ = next(splitter.split(X=X, y=y))
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
        X, _, y, _ = train_test_split(  # type: ignore - we know it returns same type
            X, y,
            train_size=sample_size,
            random_state=random_state,
        )

    return X, y

def reduce_precision(X: Union[np.ndarray, spmatrix]) -> Tuple[Union[np.ndarray, spmatrix], Type]:
    """ Reduces the precision of an arraylike

    Does not support ndarray a non-numeric type.
    Will convert a List[float] to ndarray.

    Parameters
    ----------
    X: List[float] | np.ndarray | spmatrix
        The data to reduce precision of

    safe: bool = True
        Whether to force `X.astype(np.float32)` even if `X.dtype` is not listed.

    Returns
    -------
    (List[Float] | ndarray | spmatrix, str )
        Returns the reduced data X along with the dtype it was reduced to.
    """
    bytes_to_precision_mapping = {
        4: np.float32,
        8: np.float32,
        16: np.float64,
    }

    bytes = byte_size(X.dtype)
    precision = bytes_to_precision_mapping.get(bytes, None)

    return X.astype(precision), precision

def reduce_dataset_size_if_too_large(
    X: Union[np.ndarray, spmatrix],
    y: np.ndarray,
    memory_limit: int,
    is_classification: bool,
    random_state: Union[int, np.random.RandomState] = None,
    include: List[str] = ['precision', 'subsampling'],
    multiplier: Union[float, int] = 10,
) -> Tuple[Union[spmatrix, np.ndarray], SUPPORTED_TARGET_TYPES]:
    """ Reduces the size of the dataset if it's too close to the memory limit.

    Attempts to do the following in the order:
        * Reduce precision if necessary
        * Subsample

    This retains the type of input if it's pd.DataFrame, spmatrix or ndarray.

    NOTE: limitations

        Does not precision reduce:
        *   ndarray with type != float{32,64,96,128} - Could be done using feat_type
        *   DataFrame - Could be done using feat_type

        Does not support subsampling
        *   When Dataframe as we can't easily estimate size - Could be done

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

    seed: int | RandomState = None
        The seed to use for subsampling.

    include: List[str] = ['precision', 'subsampling']
        A list of operations that are permitted to be performed to reduce
        the size of the dataset.

    multiplier: float | int = 10
        When performing reductions, satisfies the conditions that:
        * Reduce precision if `size(X) * multiplier >= memory_limit`
        * Subsample so that `size(X) * mulitplier = memory_limit` is satisfied``

    Returns
    -------
    Tuple[spmatrix | np.ndarray, np.ndarray]:
        The reduced X, y if reductions were needed
    """
    # Validation
    assert memory_limit > 0

    if 'precision' in include and X.dtype not in supported_dtypes:
            raise ValueError(f"Unsupported {X.dtype} for precision reduction")

    for operation in include:

        if operation == 'precision':
            # If `multiplier` times the dataset is too big for memory, we try
            # to reduce the precision if it's a high precision dataset
            if megabytes(X) * multiplier > memory_limit:
                X, precision = reduce_precision(X)
                warnings.warn(
                    f'Dataset too large for memory limit {memory_limit}MB, '
                    f'reduced the precision from {X.dtype} to {precision}',
                )
        elif operation == 'subsampling':
            # If the dataset is still too big such that we couldn't fit
            # `multiplier` of them in memory, we subsample such that we can
            if  memory_limit < megabytes(X) * multiplier:

                reduction_percent = float(memory_limit) / (megabytes(X) * multiplier)
                X, y = subsample(
                    X, y,
                    sample_size=reduction_percent,
                    is_classification=is_classification,
                    random_state=random_state
                )

                new_num_samples = int(reduction_percent * X.shape[0])
                warnings.warn(
                    f"Dataset too large for memory limit {memory_limit}MB, reduced"
                    f" number of samples from {X.shape[0]} to {new_num_samples}."
                )

    return X, y
