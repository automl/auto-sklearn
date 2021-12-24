import logging
import os

import arff

import numpy as np

import pytest

from scipy import sparse

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from autosklearn.pipeline.components.data_preprocessing.feature_type \
    import FeatTypeSplit
import autosklearn.metalearning.metafeatures.metafeatures as meta_features


@pytest.fixture
def sparse_data():
    tests_dir = __file__
    os.chdir(os.path.dirname(tests_dir))

    decoder = arff.ArffDecoder()
    with open(os.path.join("datasets", "dataset.arff")) as fh:
        dataset = decoder.decode(fh, encode_nominal=True)

    # -1 because the last attribute is the class
    attribute_types = [
        'numeric' if type(type_) != list else 'nominal'
        for name, type_ in dataset['attributes'][:-1]]
    categorical = {i: True if attribute == 'nominal' else False
                   for i, attribute in enumerate(attribute_types)}

    data = np.array(dataset['data'], dtype=np.float64)
    X = data[:, :-1]
    y = data[:, -1].reshape((-1,))

    # First, swap NaNs and zeros, because when converting an encoded
    # dense matrix to sparse, the values which are encoded to zero are lost
    X_sparse = X.copy()
    NaNs = ~np.isfinite(X_sparse)
    X_sparse[NaNs] = 0
    X_sparse = sparse.csr_matrix(X_sparse)

    X = X_sparse
    y = y
    mf = meta_features.metafeatures
    helpers = meta_features.helper_functions
    logger = logging.getLogger()
    # Precompute some helper functions
    helpers.set_value(
        "MissingValues",
        helpers["MissingValues"](X, y, logger, categorical),
        )
    mf.set_value(
        "NumberOfMissingValues",
        mf["NumberOfMissingValues"](X, y, logger, categorical),
        )
    helpers.set_value(
        "NumSymbols",
        helpers["NumSymbols"](X, y, logger, categorical),
        )
    helpers.set_value(
        "ClassOccurences",
        helpers["ClassOccurences"](X, y, logger),
        )
    return X, y, categorical


@pytest.fixture
def sparse_data_transformed():
    tests_dir = __file__
    os.chdir(os.path.dirname(tests_dir))

    decoder = arff.ArffDecoder()
    with open(os.path.join("datasets", "dataset.arff")) as fh:
        dataset = decoder.decode(fh, encode_nominal=True)

    # -1 because the last attribute is the class
    attribute_types = [
        'numeric' if type(type_) != list else 'nominal'
        for name, type_ in dataset['attributes'][:-1]]
    categorical = {i: True if attribute == 'nominal' else False
                   for i, attribute in enumerate(attribute_types)}

    data = np.array(dataset['data'], dtype=np.float64)
    X = data[:, :-1]
    y = data[:, -1].reshape((-1,))

    # First, swap NaNs and zeros, because when converting an encoded
    # dense matrix to sparse, the values which are encoded to zero are lost
    X_sparse = X.copy()
    NaNs = ~np.isfinite(X_sparse)
    X_sparse[NaNs] = 0
    X_sparse = sparse.csr_matrix(X_sparse)

    ohe = FeatTypeSplit(feat_type={
        col: 'categorical' if category else 'numerical'
        for col, category in categorical.items()
    })
    X_transformed = X_sparse.copy()
    X_transformed = ohe.fit_transform(X_transformed)
    imp = SimpleImputer(copy=False)
    X_transformed = imp.fit_transform(X_transformed)
    standard_scaler = StandardScaler(with_mean=False)
    X_transformed = standard_scaler.fit_transform(X_transformed)

    # Transform the array which indicates the categorical metafeatures
    number_numerical = np.sum(~np.array(list(categorical.values())))
    categorical_transformed = {i: True if i < (X_transformed.shape[1] - number_numerical) else False
                               for i in range(X_transformed.shape[1])}

    X = X_sparse
    X_transformed = X_transformed
    y = y
    mf = meta_features.metafeatures
    helpers = meta_features.helper_functions
    logger = logging.getLogger()

    # Precompute some helper functions
    helpers.set_value(
        "PCA",
        helpers["PCA"](X_transformed, y, logger),
        )
    helpers.set_value(
        "MissingValues",
        helpers["MissingValues"](X, y, logger, categorical),
        )
    mf.set_value(
        "NumberOfMissingValues",
        mf["NumberOfMissingValues"](X, y, logger, categorical),
        )
    helpers.set_value(
        "NumSymbols",
        helpers["NumSymbols"](X, y, logger, categorical),
        )
    helpers.set_value(
        "ClassOccurences",
        helpers["ClassOccurences"](X, y, logger),
        )
    helpers.set_value(
        "Skewnesses",
        helpers["Skewnesses"](X_transformed, y, logger,
                              categorical_transformed),
        )
    helpers.set_value(
        "Kurtosisses",
        helpers["Kurtosisses"](X_transformed, y, logger, categorical_transformed),
    )
    return X_transformed, y, categorical_transformed


def test_missing_values(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.helper_functions["MissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert sparse.issparse(mf.value)
    assert mf.value.shape == X.shape
    assert mf.value.dtype == bool
    assert 0 == np.sum(mf.value.data)


def test_number_of_missing_values(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["NumberOfMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert 0 == mf.value


def test_percentage_missing_values(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["PercentageOfMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert 0 == mf.value


def test_number_of_Instances_with_missing_values(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["NumberOfInstancesWithMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert 0 == mf.value


def test_percentage_of_Instances_with_missing_values(sparse_data):
    X, y, categorical = sparse_data
    meta_features.metafeatures.set_value(
        "NumberOfInstancesWithMissingValues",
        meta_features.metafeatures["NumberOfInstancesWithMissingValues"](
            X, y, logging.getLogger('Meta'), categorical))
    mf = meta_features.metafeatures["PercentageOfInstancesWithMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert pytest.approx(0) == mf.value


def test_number_of_features_with_missing_values(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["NumberOfFeaturesWithMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert 0 == mf.value


def test_percentage_of_features_with_missing_values(sparse_data):
    X, y, categorical = sparse_data
    meta_features.metafeatures.set_value(
        "NumberOfFeaturesWithMissingValues",
        meta_features.metafeatures["NumberOfFeaturesWithMissingValues"](
            X, y, logging.getLogger('Meta'), categorical))
    mf = meta_features.metafeatures["PercentageOfFeaturesWithMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert pytest.approx(0, mf.value)


def test_num_symbols(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.helper_functions["NumSymbols"](
        X, y, logging.getLogger('Meta'), categorical)

    symbol_frequency = [2, 0, 6, 0, 1, 3, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 2, 2]
    assert mf.value == symbol_frequency


def test_symbols_max(sparse_data):
    X, y, categorical = sparse_data
    # this is attribute steel
    mf = meta_features.metafeatures["SymbolsMax"](X, y, logging.getLogger('Meta'), categorical)
    assert mf.value == 6


def test_symbols_mean(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["SymbolsMean"](
        X, y, logging.getLogger('Meta'), categorical)
    # Empty looking spaces denote empty attributes
    symbol_frequency = [2, 6, 1, 3, 3, 1, 1, 2, 1, 1, 2, 2]
    assert pytest.approx(mf.value) == np.mean(symbol_frequency)


def test_symbols_std(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["SymbolsSTD"](
        X, y, logging.getLogger('Meta'), categorical)
    symbol_frequency = [2, 6, 1, 3, 3, 1, 1, 2, 1, 1, 2, 2]
    assert pytest.approx(mf.value) == np.std(symbol_frequency)


def test_symbols_sum(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.metafeatures["SymbolsSum"](
        X, y, logging.getLogger('Meta'), categorical)
    assert mf.value == 25


def test_skewnesses(sparse_data_transformed):
    X_transformed, y, categorical_transformed = sparse_data_transformed
    fixture = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.696970849903357, 0.626346013011262, 0.38099875966240554,
        1.4762248835141032, 0.07687661087633788, 0.3688979783036015
    ]
    mf = meta_features.helper_functions["Skewnesses"](X_transformed, y, logging.getLogger('Meta'))
    print(mf.value)
    print(fixture)
    np.testing.assert_allclose(mf.value, fixture)


def test_kurtosisses(sparse_data_transformed):
    fixture = [
        -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
        -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
        -3.0, -1.1005836114255763, -1.1786325509475744, -1.23879983823279,
        1.3934382644137013, -0.9768209837948336, -1.7937072296512784
    ]
    X_transformed, y, categorical_transformed = sparse_data_transformed
    mf = meta_features.helper_functions["Kurtosisses"](X_transformed, y, logging.getLogger('Meta'))
    print(mf.value)
    np.testing.assert_allclose(mf.value, fixture)


def test_pca_95percent(sparse_data_transformed):
    X_transformed, y, categorical_transformed = sparse_data_transformed
    mf = meta_features.metafeatures["PCAFractionOfComponentsFor95PercentVariance"](
        X_transformed, y, logging.getLogger('Meta'))
    assert pytest.approx(0.7741935483870968) == mf.value


def test_pca_kurtosis_first_pc(sparse_data_transformed):
    X_transformed, y, categorical_transformed = sparse_data_transformed
    mf = meta_features.metafeatures["PCAKurtosisFirstPC"](
        X_transformed, y, logging.getLogger('Meta'))
    assert pytest.approx(-0.15444516166802469) == mf.value


def test_pca_skewness_first_pc(sparse_data_transformed):
    X_transformed, y, categorical_transformed = sparse_data_transformed
    mf = meta_features.metafeatures["PCASkewnessFirstPC"](
        X_transformed, y, logging.getLogger('Meta'))
    assert pytest.approx(0.026514792083623905) == mf.value


def test_calculate_all_metafeatures(sparse_data):
    X, y, categorical = sparse_data
    mf = meta_features.calculate_all_metafeatures(
        X, y, categorical, "2", logger=logging.getLogger('Meta'))
    assert 52 == len(mf.metafeature_values)
