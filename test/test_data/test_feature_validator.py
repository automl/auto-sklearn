import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype

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
        array = np.random.randint(10, size=(100, 10)).astype('float')
        array[50, 0:5] = np.nan
        return array
    elif request.param == 'numpy_numericalonly_nan':
        array = np.random.uniform(10, size=(100, 10)).astype('float')
        array[50, 0:5] = np.nan
        # Somehow array is changed to dtype object after np.nan
        return array.astype('float')
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
            {'A': 1, 'B': 2, 'C': np.nan},
            {'A': 3, 'C': np.nan},
        ], dtype='category')
    elif request.param == 'pandas_numericalonly_nan':
        return pd.DataFrame([
            {'A': 1, 'B': 2, 'C': np.nan},
            {'A': 3, 'C': np.nan},
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
    validator.fit(input_data_featuretest, input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    elif isinstance(input_data_featuretest, list):
        assert isinstance(transformed_X, pd.DataFrame)
    else:
        assert isinstance(transformed_X, type(input_data_featuretest))
    assert np.shape(input_data_featuretest) == np.shape(transformed_X)
    assert validator._is_fitted


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
        'numpy_categoricalonly_nonan',
        'numpy_mixed_nonan',
        'numpy_categoricalonly_nan',
        'numpy_mixed_nan',
        'pandas_categoricalonly_nonan',
        'pandas_mixed_nonan',
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
    """
    validator = FeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
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
    assert validator._is_fitted


def test_featurevalidatorget_feat_type_from_columns():
    """
    Makes sure that encoded columns are returned by get_feat_type_from_columns
    whereas numerical columns are not returned
    """
    validator = FeatureValidator()

    df = pd.DataFrame([
        {'int': 1, 'float': 1.0, 'category': 'one', 'bool': True},
        {'int': 2, 'float': 2.0, 'category': 'two', 'bool': False},
    ])

    for col in df.columns:
        df[col] = df[col].astype(col)

    feature_types = validator.get_feat_type_from_columns(df)

    assert feature_types == {'int': 'numerical',
                             'float': 'numerical',
                             'category': 'categorical',
                             'bool': 'categorical'}


def test_features_unsupported_calls_are_raised():
    """
    Makes sure we raise a proper message to the user,
    when providing not supported data input or using the validator in a way that is not
    expected
    """
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r"Auto-sklearn does not support time"):
        validator.fit(
            pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
        )
    with pytest.raises(ValueError, match=r"Auto-sklearn only supports.*yet, the provided input"):
        validator.fit({'input1': 1, 'input2': 2})
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r"has unsupported dtype string"):
        validator.fit(pd.DataFrame([{'A': 1, 'B': 2}], dtype='string'))
    with pytest.raises(ValueError, match=r"The feature dimensionality of the train and test"):
        validator.fit(X_train=np.array([[1, 2, 3], [4, 5, 6]]),
                      X_test=np.array([[1, 2, 3, 4], [4, 5, 6, 7]]),
                      )
    with pytest.raises(ValueError, match=r"Cannot call transform on a validator that is not fit"):
        validator.transform(np.array([[1, 2, 3], [4, 5, 6]]))
    validator = FeatureValidator(feat_type=['Numerical'])
    with pytest.raises(ValueError, match=r"providing the option feat_type to the fit method is.*"):
        validator.fit(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))
    with pytest.raises(ValueError, match=r"feat_type does not have same number of.*"):
        validator.fit(np.array([[1, 2, 3], [4, 5, 6]]))
    validator = FeatureValidator(feat_type=[1, 2, 3])
    with pytest.raises(ValueError, match=r"feat_type must only contain strings.*"):
        validator.fit(np.array([[1, 2, 3], [4, 5, 6]]))
    validator = FeatureValidator(feat_type=['1', '2', '3'])
    with pytest.raises(ValueError, match=r"Only `Categorical` and `Numerical` are.*"):
        validator.fit(np.array([[1, 2, 3], [4, 5, 6]]))


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


# Actual checks for the features
@pytest.mark.parametrize(
    'openml_id',
    (
        40981,  # Australian
        3,  # kr-vs-kp
        1468,  # cnae-9
        40975,  # car
        40984,  # Segment
    ),
)
@pytest.mark.parametrize('train_data_type', ('numpy', 'pandas', 'list'))
@pytest.mark.parametrize('test_data_type', ('numpy', 'pandas', 'list'))
def test_featurevalidator_new_data_after_fit(openml_id,
                                             train_data_type, test_data_type):

    # List is currently not supported as infer_objects
    # cast list objects to type objects
    if train_data_type == 'list' or test_data_type == 'list':
        pytest.skip()

    validator = FeatureValidator()

    if train_data_type == 'numpy':
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=False)
    elif train_data_type == 'pandas':
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=True)
    else:
        X, y = sklearn.datasets.fetch_openml(data_id=openml_id,
                                             return_X_y=True, as_frame=True)
        X = X.values.tolist()
        y = y.values.tolist()

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    validator.fit(X_train)

    transformed_X = validator.transform(X_test)

    # Basic Checking
    if sparse.issparse(input_data_featuretest):
        assert sparse.issparse(transformed_X)
    elif isinstance(input_data_featuretest, list):
        assert isinstance(transformed_X, pd.DataFrame)
    else:
        assert isinstance(transformed_X, type(X_train))
    assert np.shape(X_test) == np.shape(transformed_X)


@pytest.mark.parametrize(
    'openml_id',
    (
        40981,  # Australian
        3,  # kr-vs-kp
        1468,  # cnae-9
        40975,  # car
        40984,  # Segment
        2,  # anneal
    ),
)
def test_list_to_dataframe(openml_id):

    X_pandas, y_pandas = sklearn.datasets.fetch_openml(data_id=openml_id,
                                                       return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X_pandas, y_pandas, random_state=1)

    X_list = X_train.values.tolist()
    validator = FeatureValidator()
    validator.fit(X_list)
    transformed_X = validator.transform(X_list)
    for i, col in enumerate(X_pandas.columns):
        if is_numeric_dtype(X_pandas[col].dtype):
            # convert dtype translates 72.0 to 72. Be robust against this!
            assert is_numeric_dtype(transformed_X[i].dtype)
        else:
            assert X_pandas[col].dtype.name == transformed_X[i].dtype.name, col

    # Also make sure that at testing time
    # this work
    transformed_X = validator.transform(X_test.values.tolist())
    for i, col in enumerate(X_pandas.columns):
        if is_numeric_dtype(X_pandas[col].dtype):
            # convert dtype translates 72.0 to 72. Be robust against this!
            assert is_numeric_dtype(transformed_X[i].dtype)
        else:
            assert X_pandas[col].dtype.name == transformed_X[i].dtype.name, col


@pytest.mark.parametrize(
    'input_data_featuretest',
    (
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
def test_sparse_output_is_csr(input_data_featuretest):
    validator = FeatureValidator()
    validator.fit(input_data_featuretest, input_data_featuretest)
    transformed_X = validator.transform(input_data_featuretest)
    assert sparse.issparse(transformed_X)
    assert isinstance(transformed_X, sparse.csr_matrix)


def test_unsupported_dataframe_sparse():
    df = pd.DataFrame({'A': pd.Series(pd.arrays.SparseArray(np.random.randn(10)))})
    validator = FeatureValidator()
    with pytest.raises(ValueError, match=r"Auto-sklearn does not yet support sparse pandas"):
        validator.fit(df)
