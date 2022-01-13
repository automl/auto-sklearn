import numpy as np

import pandas as pd

import pytest
from pandas.api.types import is_numeric_dtype

from scipy import sparse

import sklearn.datasets
import sklearn.model_selection
from sklearn.utils.multiclass import type_of_target

from autosklearn.data.target_validator import TargetValidator


# Fixtures to be used in this class. By default all elements have 100 datapoints
@pytest.fixture
def input_data_targettest(request):
    if request.param == 'series_binary':
        return pd.Series([1, -1, -1, 1])
    elif request.param == 'series_multiclass':
        return pd.Series([1, 0, 2])
    elif request.param == 'series_multilabel':
        return pd.Series([[1, 0], [0, 1]])
    elif request.param == 'series_continuous':
        return pd.Series([0.1, 0.6, 0.7])
    elif request.param == 'series_continuous-multioutput':
        return pd.Series([[1.5, 2.0], [3.0, 1.6]])
    elif request.param == 'pandas_binary':
        return pd.DataFrame([1, -1, -1, 1])
    elif request.param == 'pandas_multiclass':
        return pd.DataFrame([1, 0, 2])
    elif request.param == 'pandas_multilabel':
        return pd.DataFrame([[1, 0], [0, 1]])
    elif request.param == 'pandas_continuous':
        return pd.DataFrame([0.1, 0.6, 0.7])
    elif request.param == 'pandas_continuous-multioutput':
        return pd.DataFrame([[1.5, 2.0], [3.0, 1.6]])
    elif request.param == 'numpy_binary':
        return np.array([1, -1, -1, 1])
    elif request.param == 'numpy_multiclass':
        return np.array([1, 0, 2])
    elif request.param == 'numpy_multilabel':
        return np.array([[1, 0], [0, 1]])
    elif request.param == 'numpy_continuous':
        return np.array([0.1, 0.6, 0.7])
    elif request.param == 'numpy_continuous-multioutput':
        return np.array([[1.5, 2.0], [3.0, 1.6]])
    elif request.param == 'list_binary':
        return [1, -1, -1, 1]
    elif request.param == 'list_multiclass':
        return [1, 0, 2]
    elif request.param == 'list_multilabel':
        return [[0, 1], [1, 0]]
    elif request.param == 'list_continuous':
        return [0.1, 0.6, 0.7]
    elif request.param == 'list_continuous-multioutput':
        return [[1.5, 2.0], [3.0, 1.6]]
    elif 'openml' in request.param:
        _, openml_id = request.param.split('_')
        X, y = sklearn.datasets.fetch_openml(data_id=int(openml_id),
                                             return_X_y=True, as_frame=True)
        if len(y.shape) > 1 and y.shape[1] > 1 and np.any(y.eq('TRUE').any(1).to_numpy()):
            # This 'if' is only asserted for multi-label data
            # Force the downloaded data to be interpreted as multilabel
            y = y.dropna()
            y.replace('FALSE', 0, inplace=True)
            y.replace('TRUE', 1, inplace=True)
            y = y.astype(int)
        return y
    elif 'sparse' in request.param:
        # We expect the names to be of the type sparse_csc_nonan
        sparse_, type_, nan_ = request.param.split('_')
        if 'nonan' in nan_:
            data = np.ones(3)
        else:
            data = np.array([1, 2, np.nan])

        # Then the type of sparse
        if 'csc' in type_:
            return sparse.csc_matrix(data)
        elif 'csr' in type_:
            return sparse.csr_matrix(data)
        elif 'coo' in type_:
            return sparse.coo_matrix(data)
        elif 'bsr' in type_:
            return sparse.bsr_matrix(data)
        elif 'lil' in type_:
            return sparse.lil_matrix(data)
        elif 'dok' in type_:
            return sparse.dok_matrix(np.vstack((data, data, data)))
        elif 'dia' in type_:
            return sparse.dia_matrix(np.vstack((data, data, data)))
        else:
            ValueError("Unsupported indirect fixture {}".format(request.param))
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


# Actual checks for the targets
@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'series_multiclass',
        'series_continuous',
        'pandas_binary',
        'pandas_multiclass',
        'pandas_multilabel',
        'pandas_continuous',
        'pandas_continuous-multioutput',
        'numpy_binary',
        'numpy_multiclass',
        'numpy_multilabel',
        'numpy_continuous',
        'numpy_continuous-multioutput',
        'list_binary',
        'list_multiclass',
        'list_multilabel',
        'list_continuous',
        'list_continuous-multioutput',
        'sparse_bsr_nonan',
        'sparse_coo_nonan',
        'sparse_csc_nonan',
        'sparse_csr_nonan',
        'sparse_lil_nonan',
        'openml_204',
    ),
    indirect=True
)
def test_targetvalidator_supported_types_noclassification(input_data_targettest):
    validator = TargetValidator(is_classification=False)
    validator.fit(input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    if sparse.issparse(input_data_targettest):
        assert sparse.issparse(transformed_y)
    else:
        assert isinstance(transformed_y, np.ndarray)
    epected_shape = np.shape(input_data_targettest)
    if len(epected_shape) > 1 and epected_shape[1] == 1:
        # The target should have (N,) dimensionality instead of (N, 1)
        epected_shape = (epected_shape[0], )
    assert epected_shape == np.shape(transformed_y)
    assert np.issubdtype(transformed_y.dtype, np.number)
    assert validator._is_fitted

    # Because there is no classification, we do not expect a encoder
    assert validator.encoder is None

    if hasattr(input_data_targettest, "iloc"):
        np.testing.assert_array_equal(
            np.ravel(input_data_targettest.to_numpy()),
            np.ravel(transformed_y)
        )
    elif sparse.issparse(input_data_targettest):
        np.testing.assert_array_equal(
            np.ravel(input_data_targettest.todense()),
            np.ravel(transformed_y.todense())
        )
    else:
        np.testing.assert_array_equal(
            np.ravel(np.array(input_data_targettest)),
            np.ravel(transformed_y)
        )


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'series_multiclass',
        'pandas_binary',
        'pandas_multiclass',
        'numpy_binary',
        'numpy_multiclass',
        'list_binary',
        'list_multiclass',
        'sparse_bsr_nonan',
        'sparse_coo_nonan',
        'sparse_csc_nonan',
        'sparse_csr_nonan',
        'sparse_lil_nonan',
        'openml_2',
    ),
    indirect=True
)
def test_targetvalidator_supported_types_classification(input_data_targettest):
    validator = TargetValidator(is_classification=True)
    validator.fit(input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    if sparse.issparse(input_data_targettest):
        assert sparse.issparse(transformed_y)
    else:
        assert isinstance(transformed_y, np.ndarray)
    epected_shape = np.shape(input_data_targettest)
    if len(epected_shape) > 1 and epected_shape[1] == 1:
        # The target should have (N,) dimensionality instead of (N, 1)
        epected_shape = (epected_shape[0], )
    assert epected_shape == np.shape(transformed_y)
    assert np.issubdtype(transformed_y.dtype, np.number)
    assert validator._is_fitted

    # Because there is no classification, we do not expect a encoder
    if not sparse.issparse(input_data_targettest):
        assert validator.encoder is not None

        # The encoding should be per column
        if len(transformed_y.shape) == 1:
            assert np.min(transformed_y) == 0
            assert np.max(transformed_y) == len(np.unique(transformed_y)) - 1
        else:
            for col in range(transformed_y.shape[1]):
                assert np.min(transformed_y[:, col]) == 0
                assert np.max(transformed_y[:, col]) == len(np.unique(transformed_y[:, col])) - 1

        # Make sure we can perform inverse transform
        y_inverse = validator.inverse_transform(transformed_y)
        if hasattr(input_data_targettest, 'dtype'):
            # In case of numeric, we need to make sure dtype is preserved
            if is_numeric_dtype(input_data_targettest.dtype):
                assert y_inverse.dtype == input_data_targettest.dtype
            # Then make sure every value is properly inverse-transformed
            np.testing.assert_array_equal(np.array(y_inverse), np.array(input_data_targettest))
        elif hasattr(input_data_targettest, 'dtypes'):
            if is_numeric_dtype(input_data_targettest.dtypes[0]):
                assert y_inverse.dtype == input_data_targettest.dtypes[0]
            # Then make sure every value is properly inverse-transformed
            np.testing.assert_array_equal(np.array(y_inverse),
                                          # pandas is always (N, 1) but targets are ravel()
                                          input_data_targettest.to_numpy().reshape(-1))
    else:
        # Sparse is not encoded, mainly because the sparse data is expected
        # to be numpy of numerical type -- which currently does not require encoding
        np.testing.assert_array_equal(
            np.ravel(input_data_targettest.todense()),
            np.ravel(transformed_y.todense())
        )


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'pandas_binary',
        'numpy_binary',
        'list_binary',
        'openml_1066',
    ),
    indirect=True
)
def test_targetvalidator_binary(input_data_targettest):
    assert type_of_target(input_data_targettest) == 'binary'
    validator = TargetValidator(is_classification=True)
    # Test the X_test also!
    validator.fit(input_data_targettest, input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    assert type_of_target(transformed_y) == 'binary'


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_multiclass',
        'pandas_multiclass',
        'numpy_multiclass',
        'list_multiclass',
        'openml_54',
    ),
    indirect=True
)
def test_targetvalidator_multiclass(input_data_targettest):
    assert type_of_target(input_data_targettest) == 'multiclass'
    validator = TargetValidator(is_classification=True)
    # Test the X_test also!
    validator.fit(input_data_targettest, input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    assert type_of_target(transformed_y) == 'multiclass'


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'pandas_multilabel',
        'numpy_multilabel',
        'list_multilabel',
        'openml_40594',
    ),
    indirect=True
)
def test_targetvalidator_multilabel(input_data_targettest):
    assert type_of_target(input_data_targettest) == 'multilabel-indicator'
    validator = TargetValidator(is_classification=True)
    # Test the X_test also!
    validator.fit(input_data_targettest, input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    assert type_of_target(transformed_y) == 'multilabel-indicator'


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_continuous',
        'pandas_continuous',
        'numpy_continuous',
        'list_continuous',
        'openml_531',
    ),
    indirect=True
)
def test_targetvalidator_continuous(input_data_targettest):
    assert type_of_target(input_data_targettest) == 'continuous'
    validator = TargetValidator(is_classification=False)
    # Test the X_test also!
    validator.fit(input_data_targettest, input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    assert type_of_target(transformed_y) == 'continuous'


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'pandas_continuous-multioutput',
        'numpy_continuous-multioutput',
        'list_continuous-multioutput',
        'openml_41483',
    ),
    indirect=True
)
def test_targetvalidator_continuous_multioutput(input_data_targettest):
    assert type_of_target(input_data_targettest) == 'continuous-multioutput'
    validator = TargetValidator(is_classification=False)
    # Test the X_test also!
    validator.fit(input_data_targettest, input_data_targettest)
    transformed_y = validator.transform(input_data_targettest)
    assert type_of_target(transformed_y) == 'continuous-multioutput'


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'pandas_binary',
        'numpy_binary',
        'list_binary',
    ),
    indirect=True
)
def test_targetvalidator_fitontypeA_transformtypeB(input_data_targettest):
    """
    Check if we can fit in a given type (numpy) yet transform
    if the user changes the type (pandas then)

    This is problematic only in the case we create an encoder
    """
    validator = TargetValidator(is_classification=True)
    validator.fit(input_data_targettest)
    if isinstance(input_data_targettest, pd.DataFrame):
        complementary_type = input_data_targettest.to_numpy()
    elif isinstance(input_data_targettest, pd.Series):
        complementary_type = pd.DataFrame(input_data_targettest)
    elif isinstance(input_data_targettest, np.ndarray):
        complementary_type = pd.DataFrame(input_data_targettest)
    elif isinstance(input_data_targettest, list):
        complementary_type = pd.DataFrame(input_data_targettest)
    validator.transform(complementary_type)


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_multilabel',
        'series_continuous-multioutput',
    ),
    indirect=True
)
def test_type_of_target_unsupported(input_data_targettest):
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input
    """
    validator = TargetValidator()
    with pytest.raises(ValueError, match=r"legacy multi-.* data representation."):
        validator.fit(input_data_targettest)


def test_target_unsupported():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input
    """
    validator = TargetValidator(is_classification=True)
    with pytest.raises(ValueError, match=r"The dimensionality of the train and test targets"):
        validator.fit(
            np.array([[0, 1, 0], [0, 1, 1]]),
            np.array([[0, 1, 0, 0], [0, 1, 1, 1]]),
        )
    with pytest.raises(ValueError, match=r"Train and test targets must both have the same dtypes"):
        validator.fit(
            pd.DataFrame({'a': [1, 2, 3]}),
            pd.DataFrame({'a': [True, False, False]}),
        )
    with pytest.raises(ValueError, match=r"Provided targets are not supported.*"):
        validator.fit(
            np.array([[0, 1, 2], [0, 3, 4]]),
            np.array([[0, 1, 2, 5], [0, 3, 4, 6]]),
        )
    with pytest.raises(ValueError, match="Train and test targets must both have the same"):
        validator.fit(
            pd.DataFrame({'string': ['foo']}),
            pd.DataFrame({'int': [1]}),
        )
    with pytest.raises(ValueError, match=r"Auto-sklearn only supports Numpy arrays, .*"):
        validator.fit({'input1': 1, 'input2': 2})
    with pytest.raises(ValueError, match=r"arget values cannot contain missing/NaN values"):
        validator.fit(np.array([np.nan, 1, 2]))
    with pytest.raises(ValueError, match=r"arget values cannot contain missing/NaN values"):
        validator.fit(sparse.csr_matrix(np.array([1, 2, np.nan])))
    with pytest.raises(ValueError, match=r"Cannot call transform on a validator that is not fit"):
        validator.transform(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match=r"Cannot call inverse_transform on a validator that is"):
        validator.inverse_transform(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match=r"Multi-dimensional classification is not yet supported"):
        validator._fit(np.array([[1, 2, 3], [1, 5, 6]]))

    # Dia/ DOK are not supported as type of target makes calls len on the array
    # which causes TypeError: len() of unsized object. Basically, sparse data as
    # multi-label is the only thing that makes sense in this format.
    with pytest.raises(ValueError, match=r"The provided data could not be interpreted by Sklearn"):
        validator.fit(sparse.dia_matrix(np.array([1, 2, 3])))

    validator.fit(np.array([[0, 1, 0], [0, 1, 1]]))
    with pytest.raises(ValueError, match=r"Number of outputs changed from"):
        validator.fit(np.array([0, 1, 0]))


def test_targetvalidator_inversetransform():
    """
    Test that the encoding/decoding works in 1D
    """
    validator = TargetValidator(is_classification=True)
    validator.fit(
        pd.DataFrame(data=['a', 'a', 'b', 'c', 'a'], dtype='category'),
    )
    y = validator.transform(
        pd.DataFrame(data=['a', 'a', 'b', 'c', 'a'], dtype='category'),
    )
    np.testing.assert_array_almost_equal(np.array([0, 0, 1, 2, 0]), y)

    y_decoded = validator.inverse_transform(y)
    assert ['a', 'a', 'b', 'c', 'a'] == y_decoded.tolist()

    assert validator.classes_.tolist() == ['a', 'b', 'c']

    validator = TargetValidator(is_classification=True)
    multi_label = pd.DataFrame(
        np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
        dtype=bool
    )
    validator.fit(multi_label)
    y = validator.transform(multi_label)

    y_decoded = validator.inverse_transform(y)
    np.testing.assert_array_almost_equal(y, y_decoded)

    # Multilabel classification is not encoded
    # For this reason, classes_ attribute does not contain a class
    np.testing.assert_array_almost_equal(validator.classes_, np.array([]))


# Actual checks for the targets
@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'series_multiclass',
        'pandas_binary',
        'pandas_multiclass',
        'numpy_binary',
        'numpy_multiclass',
        'list_binary',
        'list_multiclass',
    ),
    indirect=True
)
def test_unknown_categories_in_targets(input_data_targettest):
    validator = TargetValidator(is_classification=True)
    validator.fit(input_data_targettest)

    # Add an extra category
    if isinstance(input_data_targettest, list):
        input_data_targettest.append(input_data_targettest[-1] + 5000)
    elif isinstance(input_data_targettest, (pd.DataFrame, pd.Series)):
        input_data_targettest.iloc[-1] = 5000
    elif isinstance(input_data_targettest, np.ndarray):
        input_data_targettest[-1] = 5000

    x_t = validator.transform(input_data_targettest)
    assert x_t[-1].item(0) == -1


def test_is_single_column_target():
    validator = TargetValidator(is_classification=True)
    validator.fit(np.array([1, 2, 3, 4]))
    assert validator.is_single_column_target()

    validator = TargetValidator(is_classification=True)
    validator.fit(np.array([[1, 0, 1, 0], [1, 1, 1, 1]]))
    assert not validator.is_single_column_target()
