import numpy as np
import scipy.sparse

from autosklearn.pipeline.implementations.CategoryShift import CategoryShift

import pytest


@pytest.fixture
def category_shift() -> CategoryShift:
    """Unfitted CategoryShift transformer."""
    return CategoryShift()


def test_dense(category_shift: CategoryShift) -> None:
    """
    Expects
    -------
    * Dense integer categories are shifted by 3.
    """
    X = np.random.randint(0, 255, (3, 4))
    Y = category_shift.fit_transform(X)
    assert (Y == X + 3).all()


def test_sparse(category_shift: CategoryShift) -> None:
    """
    Expects
    -------
    * Sparse integer categories are shifted by 3 while preserving sparsity structure.
    """
    X = scipy.sparse.csc_matrix(
        ([1, 2, 0, 4], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4)
    )
    Y = category_shift.fit_transform(X)
    X.data += 3
    assert (Y.todense() == X.todense()).all()
    # Check if the sparsity stays the same before and after the transformation
    assert X.data.shape == Y.data.shape
    assert (X.indices == Y.indices).all()
    assert (X.indptr == Y.indptr).all()


@pytest.mark.parametrize(
    "X",
    [
        pytest.param(np.array([[-1, 2], [3, 4]]), id="negative"),
        pytest.param(np.array([["a", "b"], ["c", "d"]]), id="string"),
    ],
)
def test_invalid_input_raises(category_shift: CategoryShift, X: np.ndarray) -> None:
    """
    Expects
    -------
    * Negative and string inputs raise ValueError.
    """
    with pytest.raises(ValueError):
        category_shift.fit_transform(X)
