from __future__ import annotations

from typing import Any

import math
import shutil
import tempfile
import uuid
from pathlib import Path

sizes = {
    "B": 0,
    "KB": 1,
    "MB": 2,
    "GB": 3,
    "TB": 4,
}


def sizeof(path: Path | str, unit: str = "B") -> float:
    """Get the size of some path object

    Parameters
    ----------
    path : Path | str
        The path of the file or directory to get the size of

    unit : "B" | "KB" | "MB" | "GB" | "TB" = "B"
        What unit to get the answer in

    Returns
    -------
    float
        The size of the folder/file in the given units
    """
    if unit not in sizes:
        raise ValueError(f"Not a known unit {unit}")

    if not isinstance(path, Path):
        path = Path(path)

    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

    power = sizes[unit]
    return size / math.pow(1024, power)


def rmtree(
    path: Path | str,
    *,
    atomic: bool = False,
    tmp: bool | Path | str = False,
    **kwargs: Any,
) -> None:
    """Delete a file or directory

    Parameters
    ----------
    path: Path | str
        The path to delete

    atomic: bool = False
        Whether to delete the file/folder atomically. This is done by first
        using a `move` before `rmtree`.

        The `move` is not guaranteed to be atomic if moving between
        different file systems which can happen when moving to /tmp,
        depending on the OS and setup.

        The deletion part is not atomic.

        * https://docs.python.org/3/library/shutil.html#shutil.move

    tmp: bool | Path | str = False
        If bool, this defines whether atomic should use the tmp dir
        for it's move. Otherwise, a path can be specified to use

    **kwargs
        Forwarded to `rmtree`
        * https://docs.python.org/3/library/shutil.html#shutil.rmtree
    """
    if isinstance(path, str):
        path = Path(path)

    if atomic:
        if tmp is True:
            dir = Path(tempfile.gettempdir())
            uid = uuid.uuid4()
            mvpath = dir / f"autosklearn-{path.name}.old_{uid}"

        elif tmp is False:
            uid = uuid.uuid4()
            mvpath = path.parent / f"{path.name}.old_{uid}"
        else:
            mvpath = tmp if isinstance(tmp, Path) else Path(tmp)

        shutil.move(str(path), str(mvpath))
        shutil.rmtree(mvpath, **kwargs)
    else:
        shutil.rmtree(path, **kwargs)
