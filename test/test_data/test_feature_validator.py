import numpy as np

import pandas as pd

import pytest

from scipy import sparse

import sklearn.datasets
import sklearn.model_selection

from autosklearn.data.feature_validator import FeatureValidator


# Fixtures to be used in this class. By default all elements have 100 datapoints
@pytest.fixture
def input_data_featuretest(request):
    if request.param == 'numpy_categoricalonly_nonan':
        return np.random.randint(10, size=(100, 10))
    elif request.param == 'numpy_numericalonly_nonan':
        return np.random.uniform(10, size=(100, 10))
    elif request.param == 'numpy_mixed_nonan':
        return np.column_stack([
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 3)),
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 1)),
        ])
    elif request.param == 'numpy_string_nonan':
        return np.array([
            ['a', 'b', 'c', 'a', 'b', 'c'],
            ['a', 'b', 'd', 'r', 'b', 'c'],
        ])
    elif request.param == 'numpy_categoricalonly_nan':
        array = np.random.randint(10, size=(100, 10))
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_numericalonly_nan':
        array = np.random.uniform(10, size=(100, 10))
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_mixed_nan':
        array = np.column_stack([
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 3)),
            np.random.uniform(10, size=(100, 3)),
            np.random.randint(10, size=(100, 1)),
        ])
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_string_nan':
        return np.array([
            ['a', 'b', 'c', 'a', 'b', 'c'],
            [np.nan, 'b', 'd', 'r', 'b', 'c'],
        ])
    elif request.param == 'pandas_categoricalonly_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='float')
    elif request.param == 'pandas_mixed_nonan':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame
    elif request.param == 'pandas_categoricalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3},
        ], dtype='float')
    elif request.param == 'pandas_mixed_nan':
        frame = pd.DataFrame([
            {'A': 1, 'B': 2, 'C': 8},
            {'A': 3, 'B': 4},
        ], dtype='category')
        frame['B'] = pd.to_numeric(frame['B'])
        return frame
    elif request.param == 'pandas_string_nonan':
        return pd.DataFrame([
            {'A': 1, 'B': 2},
            {'A': 3, 'B': 4},
        ], dtype='string')
    elif request.param == 'list_categoricalonly_nonan':
        return [
            ['a', 'b', 'c', 'd'],
            ['e', 'f', 'c', 'd'],
        ]
    elif request.param == 'list_numericalonly_nonan':
        return [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
    elif request.param == 'list_mixed_nonan':
        return [
            ['a', 2, 3, 4],
            ['b', 6, 7, 8]
        ]
    elif request.param == 'list_categoricalonly_nan':
        return [
            ['a', 'b', 'c', np.nan],
            ['e', 'f', 'c', 'd'],
        ]
    elif request.param == 'list_numericalonly_nan':
        return [
            [1, 2, 3, np.nan],
            [5, 6, 7, 8]
        ]
    elif request.param == 'list_mixed_nan':
        return [
            ['a', np.nan, 3, 4],
            ['b', 6, 7, 8]
        ]
    elif 'sparse' in request.param:
        # We expect the names to be of the type sparse_csc_nonan
        sparse_, type_, nan_ = request.param.split('_')
        if 'nonan' in nan_:
            data = np.ones(3)
        else:
            data = np.array([1, 2, np.nan])

        # Then the type of sparse
        row_ind = np.array([0, 1, 2])
        col_ind = np.array([1, 2, 1])
        if 'csc' in type_:
            return sparse.csc_matrix((data, (row_ind, col_ind)))
        elif 'csr' in type_:
            return sparse.csr_matrix((data, (row_ind, col_ind)))
        elif 'coo' in type_:
            return sparse.coo_matrix((data, (row_ind, col_ind)))
        elif 'bsr' in type_:
            return sparse.bsr_matrix((data, (row_ind, col_ind)))
        elif 'lil' in type_:
            return sparse.lil_matrix((data))
        elif 'dok' in type_:
            return sparse.dok_matrix(np.vstack((data, data, data)))
        elif 'dia' in type_:
            return sparse.dia_matrix(np.vstack((data, data, data)))
        else:
            ValueError("Unsupported indirect fixture {}".format(request.param))
    elif 'openml' in request.param:
        _, openml_id = request.param.split('_')
        X, y = sklearn.datasets.fetch_openml(data_id=int(openml_id),
                                             return_X_y=True, as_frame=True)
        return X
    else:
        ValueError("Unsupported indirect fixture {}".format(request.param))


# Actual checks for the features
@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_categoricalonly_nonan',
        'numpy_numericalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_numericalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_numericalonly_nonan',
        'pandas_mixed_nonan',
        'pandas_numericalonly_nan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'sparse_lil_nonan',
        'sparse_lil_nan',
        'openml_40981',  # Australian
    ),
    indirect=True
)
def test_featurevalidator_supported_types(input_data_featuretest):
    validator = FeatureValidator()
    validator.fit(input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    else:
        assert isinstance(transformed_X, np.ndarray)
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'list_categoricalonly_nonan',
        'list_categoricalonly_nan',
        'list_mixed_nonan',
        'list_mixed_nan',
    ),
    indirect=True
)
def test_featurevalidator_unsupported_list(input_data_featuretest):
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r".*has invalid type object. Cast it to a valid dtype.*"):
        validator.fit(input_data_featuretest)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_string_nonan',
        'numpy_string_nan',
    ),
    indirect=True
)
def test_featurevalidator_unsupported_numpy(input_data_featuretest):
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r".*When providing a numpy array.*not supported."):
        validator.fit(input_data_featuretest)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'pandas_categoricalonly_nan',
        'pandas_mixed_nan',
        'openml_179',  # adult workclass has NaN in columns
    ),
    indirect=True
)
def test_featurevalidator_unsupported_pandas(input_data_featuretest):
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r"Categorical features in a dataframe.*missing/NaN"):
        validator.fit(input_data_featuretest)


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_categoricalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'sparse_lil_nonan',
    ),
    indirect=True
)
def test_featurevalidator_fitontypeA_transformtypeB(input_data_featuretest):
    """
    Check if we can fit in a given type (numpy) yet transform
    if the user changes the type (pandas then)

    This is problematic only in the case we create an encoder
    """
    validator = FeatureValidator()
    validator.fit(input_data_featuretest)
    if isinstance(input_data_featuretest, pd.DataFrame):
        complementary_type = input_data_featuretest.to_numpy()
    elif isinstance(input_data_featuretest, np.ndarray):
        complementary_type = pd.DataFrame(input_data_featuretest)
    elif isinstance(input_data_featuretest, list):
        complementary_type = pd.DataFrame(input_data_featuretest)
    elif sparse.issparse(input_data_featuretest):
        complementary_type = sparse.csr_matrix(input_data_featuretest.todense())
    else:
        raise ValueError(type(input_data_featuretest))
    transformed_X = validator.transform(complementary_type)
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert np.issubdtype(transformed_X.dtype, np.number)
    assert validator._is_fitted


def test_featurevalidator_get_columns_to_encode():
    """
    Makes sure that encoded columns are returned by _get_columns_to_encode
    whereas numerical columns are not returned
    """
    validator = FeatureValidator()

    df = pd.DataFrame([
        {'int': 1, 'float': 1.0, 'category': 'one', 'bool': True},
        {'int': 2, 'float': 2.0, 'category': 'two', 'bool': False},
    ])

    for col in df.columns:
        df[col] = df[col].astype(col)

    enc_columns, feature_types = validator._get_columns_to_encode(df)

    assert enc_columns == ['category', 'bool']
    assert feature_types == ['numerical', 'numerical', 'categorical', 'categorical']


def test_dataframe_input_unsupported():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input
    """
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r"Auto-sklearn does not support time"):
        validator.fit(
            pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
        )
    with pytest.raises(ValueError, match="has invalid type object"):
        validator.fit(
            pd.DataFrame({'string': ['foo']})
        )
    with pytest.raises(ValueError, match=r"Auto-sklearn only supports.*yet, the provided input"):
        validator.fit({'input1': 1, 'input2': 2})
    with pytest.raises(ValueError, match=r"has unsupported dtype string"):
        validator.fit(pd.DataFrame([{'A': 1, 'B': 2}], dtype='string'))


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'numpy_numericalonly_nonan',
        'numpy_numericalonly_nan',
        'pandas_numericalonly_nonan',
        'pandas_numericalonly_nan',
        'list_numericalonly_nonan',
        'list_numericalonly_nan',
        # Category in numpy is handled via feat_type
        'numpy_categoricalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_mixed_nan',
        'sparse_bsr_nonan',
        'sparse_bsr_nan',
        'sparse_coo_nonan',
        'sparse_coo_nan',
        'sparse_csc_nonan',
        'sparse_csc_nan',
        'sparse_csr_nonan',
        'sparse_csr_nan',
        'sparse_dia_nonan',
        'sparse_dia_nan',
        'sparse_dok_nonan',
        'sparse_dok_nan',
        'sparse_lil_nonan',
        'sparse_lil_nan',
    ),
    indirect=True
)
def test_no_encoder_created(input_data_featuretest):
    """
    Makes sure that for numerical only features, no encoder is created
    """
    validator = FeatureValidator()
    validator.fit(input_data_featuretest)
    validator.transform(input_data_featuretest)
    assert validator.encoder is None


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
    ),
    indirect=True
)
def test_encoder_created(input_data_featuretest):
    """
    This test ensures an encoder is created if categorical data is provided
    """
    validator = FeatureValidator()
    validator.fit(input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    assert validator.encoder is not None

    # Make sure that the encoded features are actually encoded. Categorical columns are at
    # the start after transformation. In our fixtures, this is also honored prior encode
    enc_columns, feature_types = validator._get_columns_to_encode(input_data_featuretest)

    # At least one categorical
    assert 'categorical' in validator.feat_type

    # Numerical if the original data has numerical only columns
    if np.any([pd.api.types.is_numeric_dtype(input_data_featuretest[col]
                                             ) for col in input_data_featuretest.columns]):
        assert 'numerical' in validator.feat_type
    for i, feat_type in enumerate(feature_types):
        if 'numerical' in feat_type:
            np.testing.assert_array_equal(
                transformed_X[:, i],
                input_data_featuretest[input_data_featuretest.columns[i]].to_numpy()
            )
        elif 'categorical' in feat_type:
            np.testing.assert_array_equal(
                transformed_X[:, i],
                # Expect always 0, 1... because we use a ordinal encoder
                np.array([0, 1])
            )
        else:
            raise ValueError(feat_type)


def test_no_new_category_after_fit():
    """
    This test makes sure that we can actually pass new categories to the estimator
    without throwing an error
    """
    # Then make sure we catch categorical extra categories
    x = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, dtype='category')
    validator = FeatureValidator()
    validator.fit(x)
    x['A'] = x['A'].apply(lambda x: x*x)
    validator.transform(x)
