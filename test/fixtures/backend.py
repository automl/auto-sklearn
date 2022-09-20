from __future__ import annotations

from typing import Callable

import os
from distutils.dir_util import copy_tree
from pathlib import Path

from autosklearn.automl_common.common.utils.backend import Backend, create
from autosklearn.data.xy_data_manager import XYDataManager

from pytest import fixture

HERE = Path(__file__).parent.resolve()
DATAPATH = HERE.parent / "data"


def copy_backend(old: Backend | Path | str, new: Backend | Path | str) -> Backend:
    """Transfers a backend to a new path

    Parameters
    ----------
    old_backend: Backend | Path | str
        The backend to transfer from

    new_path: Backend | Path | str
        Where to place the new backend

    Returns
    -------
    Backend
        The new backend with the contents of the old
    """
    if isinstance(new, str):
        new_backend = create(
            temporary_directory=new,
            output_directory=None,
            prefix="auto-sklearn",
        )
    elif isinstance(new, Path):
        new_backend = create(
            temporary_directory=str(new),
            output_directory=None,
            prefix="auto-sklearn",
        )
    else:
        new_backend = new

    dst = new_backend.temporary_directory

    if isinstance(old, str):
        src = old
    elif isinstance(old, Path):
        src = str(old)
    else:
        src = old.temporary_directory

    copy_tree(src, dst)

    return new_backend


# TODO Update to return path once everything can use a path
@fixture
def tmp_dir(tmp_path: Path) -> str:
    """
    Fixtures
    --------
    tmp_path : Path
        Built in pytest fixture

    Returns
    -------
    str
        The directory as a str
    """
    return str(tmp_path)


@fixture
def make_backend(tmp_path: Path) -> Callable[..., Backend]:
    """Make a backend

    Parameters
    ----------
    path: Union[str, Path]
        The path to place the backend at

    template: Optional[Path] = None
        Setup with a pre-existing layout if not None

    Returns
    -------
    Backend
        The created backend object
    """
    # TODO redo once things use paths
    def _make(
        path: str | Path | None = None,
        template: Path | Backend | None = None,
        datamanager: XYDataManager | None = None,
    ) -> Backend:
        if template is not None and datamanager is not None:
            raise ValueError("Does not support template and datamanager")

        if path is None:
            _path = Path(tmp_path) / "backend"
        elif isinstance(path, str):
            _path = Path(path)
        else:
            _path = path

        assert not _path.exists(), "Path exists, Try passing path / 'backend'"

        if template is not None:
            backend = copy_backend(old=template, new=_path)
        else:
            backend = create(
                temporary_directory=str(_path),
                output_directory=None,
                prefix="auto-sklearn",
            )

            if datamanager is not None:
                backend.save_datamanager(datamanager)

        return backend

    return _make


@fixture(scope="function")
def backend(tmp_dir: str, make_backend: Callable) -> Backend:
    """A backend object

    Fixtures
    --------
    tmp_dir : str
        A directory to place the backend at

    make_backend : Callable
        Factory to make a backend

    Returns
    -------
    Backend
        A backend object
    """
    backend_path = os.path.join(tmp_dir, "backend")
    return make_backend(path=backend_path)
