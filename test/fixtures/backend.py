from typing import Callable, Union, Optional
from distutils.dir_util import copy_tree

import os
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
def make_backend() -> Callable[..., Backend]:
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
        path: Union[str, Path],
        template: Optional[Path] = None,
    ) -> Backend:
        _path = Path(path) if not isinstance(path, Path) else path
        assert not _path.exists(), "Try passing path / 'backend'"

        backend = create(
            temporary_directory=str(_path),
            output_directory=None,
            prefix="auto-sklearn",
        )

        if template is not None:
            assert template.exists()
            dest = Path(backend.temporary_directory)
            copy_tree(str(template), str(dest))

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
