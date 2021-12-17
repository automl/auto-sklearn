import numpy as np
from scipy import sparse

import pandas as pd
import pytest

from autosklearn.pipeline.components.data_preprocessing.imputation.categorical_imputation\
    import CategoricalImputation


@pytest.fixture
def input_data_imputation(request):
    size = (50, 20)
    X = np.array(np.random.randint(3, 10, size=size), dtype=float)
    mask = np.logical_not(np.random.randint(0, 5, size=size), dtype=bool)
    X[mask] = np.nan
    if request.param == 'numpy':
        pass
    elif request.param == 'pandas':
        X = pd.DataFrame(X)
    return X, mask


@pytest.mark.parametrize('input_data_imputation', ('numpy', 'pandas'), indirect=True)
@pytest.mark.parametrize('categorical', (True, False))
def test_default_imputation(input_data_imputation, categorical):
    """
    Makes sure that imputation works for both numerical and categorical data.
    This also has to be guaranteed for numpy and pandas like objects.
    """
    X, mask = input_data_imputation
    if categorical:
        imputation_value = 'missing_value'
        X = X.astype('str').astype('object')
        X[mask] = np.nan
    else:
        imputation_value = min(np.unique(X)) - 1

    Y = CategoricalImputation().fit_transform(X.copy())

    assert np.array_equal(Y == imputation_value, mask)
    assert np.array_equal(Y != imputation_value, ~mask)


@pytest.mark.parametrize('format_type', ('numpy', 'pandas'))
def test_nonzero_numerical_imputation(format_type):
    # First try with an array with 0 as only valid category. The imputation should
    # happen with -1
    X = np.full(fill_value=np.nan, shape=(10, 10))
    X[0, :] = 0
    if 'pandas' in format_type:
        X = pd.DataFrame(X)
    elif 'numpy' in format_type:
        pass
    else:
        pytest.fail(format_type)
    Y = CategoricalImputation().fit_transform(X.copy())
    np.testing.assert_equal(np.nan_to_num(X, nan=-1, copy=True), Y)

    # Then if there is also a -1 in the category, we expect -2 as imputation
    X = np.full(fill_value=np.nan, shape=(10, 10))
    X[0, :] = 0
    X[1, :] = -1
    if 'pandas' in format_type:
        X = pd.DataFrame(X)
    Y = CategoricalImputation().fit_transform(X.copy())
    np.testing.assert_equal(np.nan_to_num(X, nan=-2, copy=True), Y)


@pytest.mark.parametrize('input_data_imputation', ('numpy'), indirect=True)
def test_default_sparse(input_data_imputation):
    X, mask = input_data_imputation
    X = sparse.csr_matrix(X)
    Y = CategoricalImputation().fit_transform(X)
    Y = Y.todense()

    np.testing.assert_equal(Y == 0, mask)
    np.testing.assert_equal(Y != 0, ~mask)
