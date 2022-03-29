from __future__ import annotations

import math
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
