import numpy as np

import pandas as pd

import pytest

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
        return pd.Series([[1, 2], [3, 1]])
    elif request.param == 'series_continuous':
        return pd.Series([0.1, 0.6, 0.7])
    elif request.param == 'series_continuous-multioutput':
        return pd.Series([[1.5, 2.0], [3.0, 1.6]])
    elif request.param == 'pandas_binary':
        return pd.DataFrame([1, -1, -1, 1])
    elif request.param == 'pandas_multiclass':
        return pd.DataFrame([1, 0, 2])
    elif request.param == 'pandas_multilabel':
        return pd.DataFrame([[1, 2], [3, 1]])
    elif request.param == 'pandas_continuous':
        return pd.DataFrame([0.1, 0.6, 0.7])
    elif request.param == 'pandas_continuous-multioutput':
        return pd.Dataframe([[1.5, 2.0], [3.0, 1.6]])
    elif request.param == 'numpy_binary':
        return np.array([1, -1, -1, 1])
    elif request.param == 'numpy_multiclass':
        return np.array([1, 0, 2])
    elif request.param == 'numpy_multilabel':
        return np.array([[1, 2], [3, 1]])
    elif request.param == 'numpy_continuous':
        return np.array([0.1, 0.6, 0.7])
    elif request.param == 'numpy_continuous-multioutput':
        return np.array([[1.5, 2.0], [3.0, 1.6]])
    elif request.param == 'list_binary':
        return [1, -1, -1, 1]
    elif request.param == 'list_multiclass':
        return [1, 0, 2]
    elif request.param == 'list_multilabel':
        return [[1, 2], [3, 1]]
    elif request.param == 'list_continuous':
        return [0.1, 0.6, 0.7]
    elif request.param == 'list_continuous-multioutput':
        return [[1.5, 2.0], [3.0, 1.6]]
    elif 'openml' in request.param:
        _, openml_id = request.param.split('_')
        X, y = sklearn.datasets.fetch_openml(data_id=int(openml_id),
                                             return_X_y=True, as_frame=True)
        return y
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


# Actual checks for the targets
@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'series_multiclass',
        'series_multilabel',
        'series_continuous',
        'series_continuous-multioutput',
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
        'openml_2',
    ),
    indirect=True
)
def test_targetvalidator_supported_types_noclassification(input_data_targettest):
    validator = TargetValidator()
    validator.fit(input_data_targettest, is_classification=False)
    transformed_y = validator.transform(input_data_targettest)
    assert isinstance(transformed_y, np.ndarray)
    assert np.shape(input_data_targettest) == np.shape(transformed_y)
    assert np.issubdtype(transformed_y.dtype, np.number)
    assert validator._is_fitted

    # Because there is no classification, we do not expect a encoder
    assert validator.encoder is None

    if hasattr(input_data_targettest, "iloc"):
        np.testing.assert_array_equal(input_data_targettest.to_numpy(), transformed_y)
    else:
        np.testing.assert_array_equal(np.array(input_data_targettest), transformed_y)


@pytest.mark.parametrize(
    'input_data_targettest',
    (
        'series_binary',
        'series_multiclass',
        'series_multilabel',
        'pandas_binary',
        'pandas_multiclass',
        'pandas_multilabel',
        'numpy_binary',
        'numpy_multiclass',
        'numpy_multilabel',
        'list_binary',
        'list_multiclass',
        'list_multilabel',
    ),
    indirect=True
)
def test_targetvalidator_supported_types_classification(input_data_targettest):
    validator = TargetValidator()
    validator.fit(input_data_targettest, is_classification=True)
    transformed_y = validator.transform(input_data_targettest)
    assert isinstance(transformed_y, np.ndarray)
    assert np.shape(input_data_targettest) == np.shape(transformed_y)
    assert np.issubdtype(transformed_y.dtype, np.number)
    assert validator._is_fitted

    # Because there is no classification, we do not expect a encoder
    assert validator.encoder is not None

    np.testing.assert_array_equal(
        transformed_y,
        np.array(range(np.shape(input_data_targettest)[0])),
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
    validator = TargetValidator()
    validator.fit(input_data_targettest, is_classification=True)
    transformed_y = validator.transform(input_data_targettest)
    assert type_of_target(transformed_y) == 'binary'


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
    validator = TargetValidator()
    validator.fit(input_data_targettest, is_classification=True)
    if isinstance(input_data_targettest, pd.DataFrame):
        complementary_type = input_data_targettest.to_numpy()
    elif isinstance(input_data_targettest, np.ndarray):
        complementary_type = pd.DataFrame(input_data_targettest)
    validator.transform(complementary_type)


def test_target_unsupported():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input
    """
    validator = TargetValidator()
    with pytest.raises(ValueError, match=r"The dimensionality of the train and test targets"):
        validator.fit(
            np.array([[0, 1, 2]], [0, 3, 4]),
            np.array([[0, 1, 2, 5]], [0, 3, 4, 6]),
        )
    with pytest.raises(ValueError, match="Train and test targets must both have the same"):
        validator.fit(
            pd.DataFrame({'string': ['foo']}),
            pd.DataFrame({'int': [1]}),
        )
    with pytest.raises(ValueError, match=r"Auto-sklearn only supports Numpy arrays, .*"):
        validator.fit({'input1': 1, 'input2': 2})
    with pytest.raises(ValueError, match=r"arget values cannot contain missing/NaN values"):
        validator.fit(np.array(np.nan, 1, 2))


def test_targetvalidator_inversetransform():
    """
    Test that the encoding/decoding works in 1D
    """
    validator = TargetValidator()
    validator.fit(
        pd.DataFrame(data=['a', 'a', 'b', 'c', 'a'], dtype='category'),
        is_classification=True,
    )
    y = validator.transform(
        pd.DataFrame(data=['a', 'a', 'b', 'c', 'a'], dtype='category'),
    )
    np.testing.assert_array_almost_equal(np.array([0, 0, 1, 2, 0]), y)

    y_decoded = validator.inverse_transform(y)
    assert ['a', 'a', 'b', 'c', 'a'] == y_decoded.tolist()

    validator = TargetValidator()
    multi_label = pd.DataFrame(
        np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
        dtype=bool
    )
    validator.fit(multi_label, is_classification=True)
    y = validator.transform(multi_label)

    y_decoded = validator.inverse_transform(y)
    np.testing.assert_array_almost_equal(y, y_decoded)
