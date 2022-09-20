from __future__ import annotations

from typing import Callable, Hashable, Iterable, Iterator, TypeVar

from functools import reduce
from itertools import chain, cycle, islice, tee

T = TypeVar("T")


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
    itr: Iterable[T],
    where: int | Callable[[T], bool],
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
    if isinstance(where, int):
        lst = list(itr)
        return lst[:where], lst[where:]

    a = []
    itr2 = iter(itr)
    broke = False
    for x in itr2:
        if not where(x):
            a.append(x)
        else:
            broke = True
            break

    if broke:
        return a, [x] + list(itr2)
    else:
        return a, []


def split(
    lst: Iterable[T],
    by: Callable[[T], bool],
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

    Returns
    -------
    (a: list[T], b: list[T])
        a is where the func is True and b is where the func was False.
    """
    a = []
    b = []
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


def pairs(itr: Iterable[T]) -> Iterator[tuple[T, T]]:
    """An iterator over pairs of items in the iterator

    ..code:: python

        # Check if sorted
        if all(a < b for a, b in pairs(items)):
            ...

    Parameters
    ----------
    itr : Iterable[T]
        An itr of items

    Returns
    -------
    Iterable[tuple[T, T]]
        An itr of sequential pairs of the items
    """
    itr1, itr2 = tee(itr)

    # Skip first item
    _ = next(itr2)

    # Check there is a second element
    peek = next(itr2, None)
    if peek is None:
        raise ValueError("Can't create a pair from iterable with 1 item")

    # Put it back in
    itr2 = chain([peek], itr2)

    return iter((a, b) for a, b in zip(itr1, itr2))


def roundrobin(
    *iterables: Iterable[T],
    duplicates: bool = True,
    key: Callable[[T], Hashable] | None = None,
) -> Iterator[T]:
    """Performs a round robin iteration of several iterables

    Adapted from https://docs.python.org/3/library/itertools.html#recipes

    ..code:: python

        colours = ["orange", "red", "green"]
        fruits = ["apple", "banana", "orange"]

        list(roundrobin(colors, fruits))
        # ["orange", "apple", "red", "banana", "green", "orange"]

        list(roundrobin(colors, fruits, duplicates=False))
        # ["orange", "apple", "red", "banana", "green"]

    Parameters
    ----------
    *iterables: Iterable[T]
        Any amount of iterables

    duplicates: bool = True
        Whether duplicates are allowed

    key: Callable[[T], Hashable] | None = None
        A key to use when checking for duplicates

    Returns
    -------
    Iterator[T]
        A round robin iterator over the iterables passed
    """
    active_iterators = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)  # roundrobin workhorse

    # We split up the algorithm into removing duplicates and not. This removes the if
    # statement from within the loop at the cost of duplication.
    if duplicates:
        while active_iterators > 0:
            try:
                for nxt in nexts:
                    yield nxt()
            except StopIteration:
                active_iterators -= 1
                nexts = cycle(islice(nexts, active_iterators))

    else:
        seen = set()
        key = key if key is not None else lambda x: x  # Identity if None

        while active_iterators > 0:
            try:
                for nxt in nexts:
                    item = nxt()
                    identifier = key(item)

                    if identifier not in seen:
                        seen.add(identifier)
                        yield item

            except StopIteration:
                active_iterators -= 1
                nexts = cycle(islice(nexts, active_iterators))
