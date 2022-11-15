from __future__ import annotations

from typing import Any

import re
import warnings

from autosklearn.__version__ import __version__

CURRENT_VERSION = tuple(map(int, re.findall(r"\d+", __version__)))


def deprecate(
    active: Any | bool | None,
    msg: str,
    *,
    new: str | None = None,
    fix: str | None = None,
    ignored: str | bool = False,
    when: tuple[int, ...] | None = None,
) -> bool:
    """Set a deprecation warning whichs triggers an Error once reached

    Parameters
    ----------
    active: Any | bool | None
        A condition or not Null to check if it should be triggered

    msg : str
        Message to raise

    new: str | None = None
        If there is something new to do instead.

    fix: str | None = None
        If a fix is provided, this will be added as additional message context

    ignored: bool = False
        Specify whether the deprecated thing will be ignored

    when: tuple[int, ...] | None = None
        When to deprecate this

    Return
    ------
    bool
        Whether or not this deprecation was activated

    Raises
    ------
    RuntimeError
        Raise if `when` is passed and the deprecation time has been reached by the
        current version.
    """
    assert not (ignored and fix)

    if active is None or active is False:
        return False

    m = msg
    if isinstance(ignored, str):
        m += f"\n - {ignored} has been ignored."
    elif ignored is True:
        m += "\n - This has been ignored."

    if fix:
        m += f"\n - {fix}"

    if new:
        m += f"\n {new}"

    warnings.warn(m, DeprecationWarning, stacklevel=3)
    if when and CURRENT_VERSION >= when:
        raise RuntimeError(f"Deprecation reached: {when=} > {CURRENT_VERSION=}\n{m}")

    return True


def deprecated(active: Any | bool | None, msg: str, *, since: tuple[int, ...]) -> None:
    """Mark something as having been deprecated

    This is most notably used when there's no possiblity to do a soft-deprecation.

    Parameters
    ----------
    active: Any | bool | None
        A condition or not Null to check if it should be triggered

    msg : str
        Message to raise

    since : tuple[int, ...]
        Since when was this deprecated

    Returns
    -------
    NoReturn
    """
    if active is None or active is False:
        return

    s = ".".join(map(str, since))
    raise RuntimeError(f"since ({s}): {msg}")
