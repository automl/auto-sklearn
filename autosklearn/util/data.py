# -*- encoding: utf-8 -*-
# Functions performing various data conversions for the ChaLearn AutoML
# challenge
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from sklearn.model_selection import train_test_split

from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit

__all__ = [
    'predict_RAM_usage',
    'convert_to_num',
    'convert_to_bin'
]


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


def reduce_dataset_size_if_too_large(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    memory_limit: int,
    include: List[str] = ['precision', 'subsampling'],
    multiplier: Union[float, int] = 10,
    is_classification: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """ Reduces the size of the dataset if it's too close to the memory limit.

    Attempts to do the following in the order:
        * Reduce precision if necessary
        * Subsample

    Relies on the input being of type np.ndarray for information

    Parameters
    ----------
    X: np.ndarray
        The features of the dataset.

    y: np.ndarray
        The labels of the dataset.

    seed: int
        The seed to use for subsampling.

    memory_limit: int
        The amount of memory allocated in megabytes

    include: List[str] = ['precision', 'subsampling']
        A list of operations that are permitted to be performed to reduce
        the size of the dataset.

    multiplier: float | int
        When performing reductions, satisfies the conditions that:
        * Reduce precision if `size(X) * multiplier >= memory_limit`
        * Subsample so that `size(X) * mulitplier = memory_limit` is satisfied``

    is_classification: bool
        Whether it's a classificaiton dataset or not. This is important when
        considering how to subsample.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The reduced X, y if reductions were needed
    """

    def byte_size(dtype: np.dtype) -> int:
        """ Returns the amount of bits required for an element of dtype """
        if dtype == np.float32:
            return 4
        elif dtype in (np.float64, float):
            return 8
        elif (
            (hasattr(np, 'float128') and dtype == np.float128)
            or (hasattr(np, 'float96') and dtype == np.float96)
        ):
            # In spite of the names, np.float96 and np.float128
            # provide only as much precision as np.longdouble,
            # that is, 80 bits on most x86 machines and 64 bits
            # in standard Windows builds.
            return 16
        else:
            # Just assuming some value - very unlikely
            warnings.warn(
                f'Unknown dtype for X: {dtype}, assuming it takes 8 bit/number'
            )
            return 8

    def megabytes(X_: np.ndarray) -> float:
        """ Estimate how large X is in megabytes """
        return X_.shape[0] * X_.shape[1] * byte_size(X_.dtype) * 1e-6

    def reduce_precision(X_: np.ndarray) -> Tuple[np.ndarray, str]:
        """ Reduces the precision of a dataset, only works for X.dtype > np.float32 """
        if X_.dtype == np.float32:
            return X_, str(np.float32)

        precision_mapping = {
            4: np.float32,
            8: np.float32,
            16: np.float64,
        }
        precision = precision_mapping.get(byte_size(X_.dtype), np.float32)

        return X_.astype(precision), str(precision)

    def subsample(
        X_: np.ndarray,
        y_: np.ndarray,
        sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Subsamples the array so it fits into the memory limit """
        if is_classification:
            splitter = CustomStratifiedShuffleSplit(
                train_size=sample_size,
                random_state=seed
            )
            left_idxs, right_idxs = next(splitter.split(X=X_, y=y_))
            X_ = X_[left_idxs]
            y_ = y_[left_idxs]

        else:
            X_, _, y_, _ = train_test_split(
                X_, y_,
                train_size=sample_size,
                random_state=seed,
            )  # type: ignore

        return X_, y_

    if 'subsampling' in include:
        assert is_classification is not None

    # There is no memory limit, we can't decide how to reduce
    if not memory_limit:
        return X, y

    # If 10 times the dataset is too big for memory, we try to reduce the
    # precision if it's a high precision dataset
    if megabytes(X) * multiplier > memory_limit:
        X, precision = reduce_precision(X)
        warnings.warn(
            f'Dataset too large for memory limit {memory_limit}MB, '
            f'reduced the precision from {X.dtype} to {precision}',
        )

    # If the dataset is still too big such that we couldn't fit 10 of them in
    # memory, we subsample such that we can
    if megabytes(X) * multiplier > memory_limit:
        reduction_factor = float(memory_limit) / (megabytes(X) * multiplier)
        new_num_samples = int(reduction_factor * len(X))
        X, y = subsample(X, y, new_num_samples)
        warnings.warn(
            f"Dataset too large for memory limit {memory_limit}MB, reduced"
            f" number of samples from {X.shape[0]} to {new_num_samples}."
        )

    return X, y
