from __future__ import annotations

from typing import Callable

import os
from distutils.dir_util import copy_tree
from pathlib import Path

from autosklearn.automl_common.common.utils.backend import Backend, create

from pytest import fixture

HERE = Path(__file__).parent.resolve()
DATAPATH = HERE.parent / "data"


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
    ) -> Backend:
        if path is None:
            path = tmp_path / "backend"

        _path = Path(path) if not isinstance(path, Path) else path
        assert not _path.exists(), "Try passing path / 'backend'"

        backend = create(
            temporary_directory=str(_path),
            output_directory=None,
            prefix="auto-sklearn",
        )

        if template is not None:
            dest = Path(backend.temporary_directory)

            if isinstance(template, Backend):
                template = Path(template.temporary_directory)

            if isinstance(template, Path):
                assert template.exists()
                copy_tree(str(template), str(dest))

            else:
                raise NotImplementedError(template)

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
