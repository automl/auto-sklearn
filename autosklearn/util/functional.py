from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar, no_type_check

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


def cut(
    lst: Iterable[T],
    at: int | Callable[[T], bool],
) -> tuple[list[T], list[T]]:
    """Cut a list in two at a given index or predicate

    Parameters
    ----------
    lst : Iterable[T]
        An iterable of items

    at : int | Callable[[T], bool]
        Where to split at, either an index or a predicate

    Returns
    -------
    tuple[list[T], list[T]]
        The split items
    """
    if isinstance(at, int):
        lst = list(lst)
        return lst[:at], lst[at:]
    else:
        a = []
        itr = iter(lst)
        for x in itr:
            if not at(x):
                a.append(x)
                break

        return a, [x] + list(itr)


def split_by(
    lst: Iterable[T],
    by: Callable[[T], bool],
    *,
    split_at_first: bool = False,
) -> tuple[list[T], list[T]]:
    """Split a list in two based on a predicate.

    Note
    ----
    First element can not contain None

    Parameters
    ----------
    lst : Iterable[T]
        The iterator to split

    by : Callable[[T], bool]
        The predicate to split it on

    split_at_first: bool = False
        Whether to split at the first occurence of `func == True`

    Returns
    -------
    (a: list[T], b: list[T])
        a is where the func is True and b is where the func was False. If using
        `split_at_first = True`, b contains everything after the first
        False occurence.
    """
    a = []
    b = []
    if split_at_first:
        itr = iter(lst)
        for x in itr:
            if by(x):
                a.append(x)
            else:
                break

        return a, list(itr)  # Convert remaining to list

    else:
        for x in lst:
            if by(x):
                a.append(x)
            else:
                b.append(x)

        return a, b


def bound(val: float, bounds: tuple[float, float]) -> float:
    """Bounds a value between a low and high

    .. code:: python

        x = bound(14, low=0, high=13.1)
        # x == 13.1

    Parameters
    ----------
    val : float
        The value to bound

    bounds: tuple[foat, float]
        The bounds to bound the value between (low, high)

    Returns
    -------
    float
        The bounded value
    """
    return max(bounds[0], min(val, bounds[1]))


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
        The first index where func was True
    """
    return next((i for i, t in enumerate(itr) if func(t)), default)


@no_type_check
def value_split(
    lst: Sequence[T],
    *,
    key: Callable[[T], float] | None = None,
    low: float | None = None,
    high: float | None = None,
    at: float = 0.5,
    sort: bool = True,
) -> tuple[list[T], list[T]]:
    """Split a list according to it's values at a certain percentage.

    Will attempt to sort the values unless specified that it should not `sort`.
    The endpoints `low` and `high` are assumed to be the min and max of the sorted
    `lst`.

    The value used for splitting is calculated by

        (1 - `at`) * low + `at` * high

    ..code:: python

        # min    low           at=0.75  high/max
        #  |-----|----------------|---------|
        #  0    20               80        100
        #
        #  [----------------------][++++++++]
        #           split 1          split 2

        x = np.linspace(0, 100, 21)
        # [0, 5, 10, ..., 95, 100]

        lower, higher = value_split(x, at=0.6, low=20)

        print(lower, higher)
        # [0, 5, 10, ..., 75] [80, ..., 100]

    Parameters
    ----------
    lst : Sequence[T]
        The list of items to split

    key : Callable[[T], float] | None = None
        An optional key to access the values by

    low : float | None = None
        The lowest value to consider, otherwise will use the minimum in lst

    high : float | None = None
        The highest value to consider, otherwise will use the maximum in lst

    at : float = 0.5
        At what perecentage to split at

    sort : bool = True
        Whether to sort the values, set to False if values are sorted before hand

    Returns
    -------
    tuple[list[T], list[T]]
        The lower and upper parts of the list based on the split
    """
    if sort:
        lst = sorted(lst) if key is None else sorted(lst, key=key)

    if low is None:
        low = lst[0] if key is None else key(lst[0])

    if high is None:
        high = lst[-1] if key is None else key(lst[-1])

    # Convex combination of two points
    pivot_value = (1 - at) * low + (at) * high

    if key is None:
        greater_than_pivot = lambda x: x >= pivot_value
    else:
        greater_than_pivot = lambda x: key(x) >= pivot_value

    pivot_idx = findwhere(lst, greater_than_pivot, default=len(lst))

    return lst[:pivot_idx], lst[pivot_idx:]
