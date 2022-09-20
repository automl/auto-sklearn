from typing import Any

from pytest import mark, param


def fails(arg: Any, reason: str = "No reason given") -> Any:
    """Mark a parameter for pytest parametrize that it should fail

    ..code:: python

        @parametrize("number", [2, 3, fails(5, "some reason")])

    Parameters
    ----------
    arg : Any
        The arg that should fail

    reason : str = "No reason given"
        The reason for the expected fail

    Returns
    -------
    Any
        The param object
    """
    return param(arg, marks=mark.xfail(reason=reason))


def skip(arg: Any, reason: str = "No reason given") -> Any:
    """Mark a parameter for pytest parametrize that should be skipped

    ..code:: python

        @parametrize("number", [2, 3, skip(5, "some reason")])

    Parameters
    ----------
    arg : Any
        The arg that should be skipped

    reason : str = "No Reason given"
        The reason for skipping it

    Returns
    -------
    Any
        The param object
    """
    return param(arg, marks=mark.skip(reason=reason))
