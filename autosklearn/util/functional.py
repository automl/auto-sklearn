from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar

from functools import reduce

import numpy as np

T = TypeVar("T")


def normalize(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalizes an array along an axis

    Note
    ----
    TODO: Only works for positive numbers

    ..code:: python

        x = np.ndarray([
            [1, 1, 1],
            [2, 2, 2],
            [7, 7, 7],
        ])

        print(normalize(x, axis=0))

        np.ndarray([
            [.1, .1, .1]
            [.2, .2, .2]
            [.7, .7, .7]
        ])

        print(normalize(x, axis=1))

        np.ndarray([
            [.333, .333, .333]
            [.333, .333, .333]
            [.333, .333, .333]
        ])

    Note
    ----
    Does not account for 0 sums along an axis

    Parameters
    ----------
    x : np.ndarray
        The array to normalize

    axis : Optional[int] = None
        The axis to normalize across

    Returns
    -------
    np.ndarray
        The normalized array
    """
    return x / x.sum(axis=axis, keepdims=True)


def intersection(*items: Iterable[T]) -> set[T]:
    """Does an intersection over all collection of items

    ..code:: python

        ans = intersection(["a", "b", "c"], "ab", ("b", "c"))

        items = [(1, 2, 3), (2, 3), (4, 5)]
        ans = intesection(*items)

    Parameters
    ----------
    *items : Iterable[T]
        A list of lists

    Returns
    -------
    Set[T]
        The intersection of all items
    """
    if len(items) == 0:
        return set()

    return set(reduce(lambda s1, s2: set(s1) & set(s2), items, items[0]))


def itersplit(lst: Iterable[T], func: Callable[[T], bool]) -> tuple[list[T], list[T]]:
    """Split a list in two based on a predicate

    Parameters
    ----------
    lst : Iterable[T]
        The list to split

    func : Callable[[T], bool]
        The predicate to split it on

    Returns
    -------
    (a: list[T], b: list[T])
        Everything in a satisfies the func while nothing in b does
    """
    a = []
    b = []
    for x in lst:
        if func(x):
            a.append(x)
        else:
            b.append(x)

    return a, b


def bound(val: float, *, low: float, high: float) -> float:
    """Bounds a value between a low and high

    .. code:: python

        x = bound(14, low=0, high=13.1)
        # x == 13.1

    Parameters
    ----------
    val : float
        The value to bound

    low : float
        The low to bound against

    high : float
        The high to bound against

    Returns
    -------
    float
        The bounded value
    """
    return max(low, min(val, high))


def findwhere(itr: Iterable[T], func: Callable[[T], bool], *, default: int = -1) -> int:
    """Find the index of the next occurence where func is True.

    Parameters
    ----------
    itr : Iterable[T]
        The iterable to search over

    func : Callable[[T], bool]
        The function to use

    default : int = -1
        The default value to give if no value was found where func was True

    Returns
    -------
    int
        The index where func was True
    """
    return next((i for i, t in enumerate(itr) if func(t)), default)


def value_split(
    lst: Sequence[float],
    *,
    low: float | None = None,
    high: float | None = None,
    at: float = 0.5,
    sort: bool = True,
) -> tuple[list[float], list[float]]:
    if sort:
        lst = sorted(lst)

    if low is None:
        low = lst[0]

    if high is None:
        high = lst[-1]


