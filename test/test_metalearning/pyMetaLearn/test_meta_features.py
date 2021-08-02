import logging
import os
import tempfile
import unittest

import pandas as pd

import pytest

import arff
from joblib import Memory
import numpy as np
from sklearn.datasets import make_multilabel_classification, fetch_openml

from autosklearn.pipeline.components.data_preprocessing.feature_type \
    import FeatTypeSplit
from autosklearn.metalearning.metafeatures.metafeature import MetaFeatureValue
import autosklearn.metalearning.metafeatures.metafeatures as meta_features


@pytest.fixture(
    scope='class',
    params=('pandas', 'numpy')
)
def multilabel_train_data(request):
    cache = Memory(location=tempfile.gettempdir())
    cached_func = cache.cache(make_multilabel_classification)
    X, y = cached_func(
        n_samples=100,
        n_features=10,
        n_classes=5,
        n_labels=5,
        return_indicator=True,
        random_state=1
    )
    if request.param == 'numpy':
        return X, y
    elif request.param == 'pandas':
        return pd.DataFrame(X), y
    else:
        raise ValueError(request.param)


@pytest.fixture(
    scope='class',
    params=('pandas', 'numpy')
)
def meta_train_data(request):
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

    logger = logging.getLogger('Meta')
    meta_features.helper_functions.set_value(
        "MissingValues", meta_features.helper_functions["MissingValues"](X, y, logger, categorical),
        )
    meta_features.helper_functions.set_value(
        "NumSymbols",
        meta_features.helper_functions["NumSymbols"](X, y, logger,  categorical),
    )
    meta_features.helper_functions.set_value(
        "ClassOccurences",
        meta_features.helper_functions["ClassOccurences"](X, y, logger),
    )
    if request.param == 'numpy':
        return X, y, categorical
    elif request.param == 'pandas':
        return pd.DataFrame(X), y, categorical
    else:
        raise ValueError(request.param)


@pytest.fixture(
    scope='class',
    params=('pandas', 'numpy')
)
def meta_train_data_transformed(request):
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

    logger = logging.getLogger('Meta')
    meta_features.helper_functions.set_value(
        "MissingValues", meta_features.helper_functions["MissingValues"](X, y, logger, categorical),
        )
    meta_features.helper_functions.set_value(
        "NumSymbols",
        meta_features.helper_functions["NumSymbols"](X, y, logger,  categorical),
    )
    meta_features.helper_functions.set_value(
        "ClassOccurences",
        meta_features.helper_functions["ClassOccurences"](X, y, logger),
    )

    DPP = FeatTypeSplit(feat_type={
        col: 'categorical' if category else 'numerical' for col, category in categorical.items()
    })
    X_transformed = DPP.fit_transform(X)

    number_numerical = np.sum(~np.array(list(categorical.values())))
    categorical_transformed = {i: True if i < (X_transformed.shape[1] - number_numerical) else False
                               for i in range(X_transformed.shape[1])}

    # pre-compute values for transformed inputs
    meta_features.helper_functions.set_value(
        "PCA", meta_features.helper_functions["PCA"](X_transformed, y, logger),
    )
    meta_features.helper_functions.set_value(
        "Skewnesses", meta_features.helper_functions["Skewnesses"](
            X_transformed, y, logger, categorical_transformed),
    )
    meta_features.helper_functions.set_value(
        "Kurtosisses", meta_features.helper_functions["Kurtosisses"](
            X_transformed, y, logger, categorical_transformed)
    )

    if request.param == 'numpy':
        return X_transformed, y, categorical_transformed
    elif request.param == 'pandas':
        return pd.DataFrame(X_transformed), y, categorical_transformed
    else:
        raise ValueError(request.param)


def test_number_of_instance(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfInstances"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 898
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_classes(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfClasses"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 5
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_features(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfFeatures"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 38
    assert isinstance(mf, MetaFeatureValue)


def test_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.helper_functions["MissingValues"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert isinstance(mf.value, pd.DataFrame if hasattr(X, 'iloc') else np.ndarray)
    assert mf.value.shape == X.shape
    assert 22175 == np.count_nonzero(mf.value)


def test_number_of_Instances_with_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfInstancesWithMissingValues"](
        X, y, logging.getLogger('Meta'), categorical)
    assert mf.value == 898
    assert isinstance(mf, MetaFeatureValue)


def test_percentage_of_Instances_with_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    meta_features.metafeatures.set_value(
        "NumberOfInstancesWithMissingValues",
        meta_features.metafeatures["NumberOfInstancesWithMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical),
        )
    mf = meta_features.metafeatures["PercentageOfInstancesWithMissingValues"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == 1.0
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_features_with_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfFeaturesWithMissingValues"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 29
    assert isinstance(mf, MetaFeatureValue)


def test_percentage_of_features_with_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    meta_features.metafeatures.set_value(
        "NumberOfFeaturesWithMissingValues",
        meta_features.metafeatures["NumberOfFeaturesWithMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical))
    mf = meta_features.metafeatures["PercentageOfFeaturesWithMissingValues"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == float(29)/float(38)
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    np.save('/tmp/debug', X)
    mf = meta_features.metafeatures["NumberOfMissingValues"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 22175
    assert isinstance(mf, MetaFeatureValue)


def test_percentage_missing_values(meta_train_data):
    X, y, categorical = meta_train_data
    meta_features.metafeatures.set_value(
        "NumberOfMissingValues", meta_features.metafeatures["NumberOfMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical))
    mf = meta_features.metafeatures["PercentageOfMissingValues"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == (float(22175)/float(38*898))
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_numeric_features(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfNumericFeatures"](
        X, y, logging.getLogger('Meta'), categorical)
    assert mf.value == 6
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_categorical_features(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["NumberOfCategoricalFeatures"](
        X, y, logging.getLogger('Meta'), categorical)
    assert mf.value == 32
    assert isinstance(mf, MetaFeatureValue)


def test_ratio_numerical_to_categorical(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["RatioNumericalToNominal"](
        X, y, logging.getLogger('Meta'), categorical)
    assert pytest.approx(mf.value) == (float(6)/float(32))
    assert isinstance(mf, MetaFeatureValue)


def test_ratio_categorical_to_numerical(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["RatioNominalToNumerical"](
        X, y, logging.getLogger('Meta'), categorical)
    assert pytest.approx(mf.value) == (float(32)/float(6))
    assert isinstance(mf, MetaFeatureValue)


def test_dataset_ratio(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["DatasetRatio"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == (float(38)/float(898))
    assert isinstance(mf, MetaFeatureValue)


def test_inverse_dataset_ratio(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["InverseDatasetRatio"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == (float(898)/float(38))
    assert isinstance(mf, MetaFeatureValue)


def test_class_occurences(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.helper_functions["ClassOccurences"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == {0.0: 8.0, 1.0: 99.0, 2.0: 684.0, 4.0: 67.0, 5.0: 40.0}


def test_class_probability_min(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["ClassProbabilityMin"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == (float(8)/float(898))
    assert isinstance(mf, MetaFeatureValue)


def test_class_probability_max(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["ClassProbabilityMax"](
        X, y, logging.getLogger('Meta'),  categorical)
    assert pytest.approx(mf.value) == (float(684)/float(898))
    assert isinstance(mf, MetaFeatureValue)


def test_class_probability_mean(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["ClassProbabilityMean"](
        X, y, logging.getLogger('Meta'),  categorical)
    classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
    prob_mean = (classes / float(898)).mean()
    assert pytest.approx(mf.value) == prob_mean
    assert isinstance(mf, MetaFeatureValue)


def test_class_probability_std(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["ClassProbabilitySTD"](
        X, y, logging.getLogger('Meta'),  categorical)
    classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
    prob_std = (classes / float(898)).std()
    assert pytest.approx(mf.value) == prob_std
    assert isinstance(mf, MetaFeatureValue)


def test_num_symbols(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.helper_functions["NumSymbols"](
        X, y, logging.getLogger('Meta'),  categorical)
    symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, 0,
                        1, 1, 1, 0, 1, 1, 0, 3, 1, 0, 0, 0, 2, 2, 3, 2]
    assert mf.value == symbol_frequency


def test_symbols_min(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["SymbolsMin"](X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 1


def test_symbols_max(meta_train_data):
    X, y, categorical = meta_train_data
    # this is attribute steel
    mf = meta_features.metafeatures["SymbolsMax"](X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 7


def test_symbols_mean(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["SymbolsMean"](
        X, y, logging.getLogger('Meta'),  categorical)
    # Empty looking spaces denote empty attributes
    symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1,  #
                        1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
    assert pytest.approx(mf.value) == np.mean(symbol_frequency)


def test_symbols_std(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["SymbolsSTD"](X, y, logging.getLogger('Meta'),  categorical)
    symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1,  #
                        1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
    assert pytest.approx(mf.value) == np.std(symbol_frequency)


def test_symbols_sum(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["SymbolsSum"](X, y, logging.getLogger('Meta'),  categorical)
    assert mf.value == 49


def test_class_entropy(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.metafeatures["ClassEntropy"](
        X, y, logging.getLogger('Meta'),  categorical)
    classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
    classes = classes / sum(classes)
    entropy = -np.sum([c * np.log2(c) for c in classes])

    assert pytest.approx(mf.value) == entropy


def test_calculate_all_metafeatures(meta_train_data):
    X, y, categorical = meta_train_data
    mf = meta_features.calculate_all_metafeatures(
        X, y, categorical, "2", logger=logging.getLogger('Meta'))
    assert 52 == len(mf.metafeature_values)
    assert mf.metafeature_values['NumberOfCategoricalFeatures'].value == 32


def test_kurtosisses(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    mf = meta_features.helper_functions["Kurtosisses"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)
    assert 6 == len(mf.value)


def test_kurtosis_min(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["KurtosisMin"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_kurtosis_max(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["KurtosisMax"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_kurtosis_mean(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["KurtosisMean"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_kurtosis_std(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["KurtosisSTD"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_skewnesses(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    mf = meta_features.helper_functions["Skewnesses"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)
    assert 6 == len(mf.value)


def test_skewness_min(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["SkewnessMin"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_skewness_max(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["SkewnessMax"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_skewness_mean(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["SkewnessMean"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_skewness_std(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["SkewnessSTD"](
        X_transformed, y, logging.getLogger('Meta'), categorical_transformed)


def test_landmark_lda(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["LandmarkLDA"](X_transformed, y, logging.getLogger('Meta'))


def test_landmark_naive_bayes(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["LandmarkNaiveBayes"](
        X_transformed, y, logging.getLogger('Meta'))


def test_landmark_decision_tree(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["LandmarkDecisionTree"](
        X_transformed, y, logging.getLogger('Meta'))


def test_decision_node(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["LandmarkDecisionNodeLearner"](
        X_transformed, y, logging.getLogger('Meta'))


def test_random_node(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["LandmarkRandomNodeLearner"](
        X_transformed, y, logging.getLogger('Meta'))


@unittest.skip("Currently not implemented!")
def test_worst_node(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["LandmarkWorstNodeLearner"](
        X_transformed, y, logging.getLogger('Meta'))


def test_1NN(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    # TODO: somehow compute the expected output?
    meta_features.metafeatures["Landmark1NN"](X_transformed, y, logging.getLogger('Meta'))


def test_pca(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    meta_features.helper_functions["PCA"](X_transformed, y, logging.getLogger('Meta'))


def test_pca_95percent(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    mf = meta_features.metafeatures["PCAFractionOfComponentsFor95PercentVariance"](
        X_transformed, y, logging.getLogger('Meta'))
    assert pytest.approx(0.2716049382716049) == mf.value


def test_pca_kurtosis_first_pc(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    mf = meta_features.metafeatures["PCAKurtosisFirstPC"](
        X_transformed, y, logging.getLogger('Meta'))
    assert pytest.approx(-0.702850) != mf.value


def test_pca_skewness_first_pc(meta_train_data_transformed):
    X_transformed, y, categorical_transformed = meta_train_data_transformed
    mf = meta_features.metafeatures["PCASkewnessFirstPC"](
        X_transformed, y, logging.getLogger('Meta'))
    assert pytest.approx(0.051210) != mf.value


def test_class_occurences_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.helper_functions["ClassOccurences"](X, y, logging.getLogger('Meta'))
    assert mf.value == [{0: 16.0, 1: 84.0},
                        {0: 8.0, 1: 92.0},
                        {0: 68.0, 1: 32.0},
                        {0: 15.0, 1: 85.0},
                        {0: 28.0, 1: 72.0}]


def test_class_probability_min_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    meta_features.helper_functions.set_value(
        "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
            X, y, logging.getLogger('Meta')))
    mf = meta_features.metafeatures["ClassProbabilityMin"](X, y, logging.getLogger('Meta'))
    assert pytest.approx(mf.value) == (float(8) / float(100))
    assert isinstance(mf, MetaFeatureValue)


def test_class_probability_max_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    meta_features.helper_functions.set_value(
        "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
            X, y, logging.getLogger('Meta')))
    mf = meta_features.metafeatures["ClassProbabilityMax"](X, y, logging.getLogger('Meta'))
    assert pytest.approx(mf.value) == (float(92) / float(100))
    assert isinstance(mf, MetaFeatureValue)


def test_class_probability_mean_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    meta_features.helper_functions.set_value(
        "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
            X, y, logging.getLogger('Meta')))
    mf = meta_features.metafeatures["ClassProbabilityMean"](X, y, logging.getLogger('Meta'))
    classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
    probas = np.mean([np.mean(np.array(cls_)) / 100 for cls_ in classes])
    assert mf.value == pytest.approx(probas)
    assert isinstance(mf, MetaFeatureValue)


def test_number_of_classes_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["NumberOfClasses"](X, y, logging.getLogger('Meta'))
    assert mf.value == 5
    assert isinstance(mf, MetaFeatureValue)


def test_class_probability_std_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    meta_features.helper_functions.set_value(
        "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
            X, y, logging.getLogger('Meta')))
    mf = meta_features.metafeatures["ClassProbabilitySTD"](X, y, logging.getLogger('Meta'))
    classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
    probas = np.mean([np.std(np.array(cls_) / 100.) for cls_ in classes])
    assert pytest.approx(mf.value) == probas
    assert isinstance(mf, MetaFeatureValue)


def test_class_entropy_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["ClassEntropy"](X, y, logging.getLogger('Meta'))

    classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
    entropies = []
    for cls in classes:
        cls = np.array(cls, dtype=np.float32)
        cls = cls / sum(cls)
        entropy = -np.sum([c * np.log2(c) for c in cls])
        entropies.append(entropy)

    assert pytest.approx(mf.value) == np.mean(entropies)


def test_landmark_lda_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["LandmarkLDA"](X, y, logging.getLogger('Meta'))
    assert np.isfinite(mf.value)


def test_landmark_naive_bayes_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["LandmarkNaiveBayes"](X, y, logging.getLogger('Meta'))
    assert np.isfinite(mf.value)


def test_landmark_decision_tree_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["LandmarkDecisionTree"](X, y, logging.getLogger('Meta'))
    assert np.isfinite(mf.value)


def test_landmark_decision_node_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["LandmarkDecisionNodeLearner"](
        X, y, logging.getLogger('Meta'))
    assert np.isfinite(mf.value)


def test_landmark_random_node_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["LandmarkRandomNodeLearner"](
        X, y, logging.getLogger('Meta'))
    assert np.isfinite(mf.value)


def test_1NN_multilabel(multilabel_train_data):
    X, y = multilabel_train_data
    mf = meta_features.metafeatures["Landmark1NN"](X, y, logging.getLogger('TestMeta'))
    assert np.isfinite(mf.value)


def test_calculate_all_metafeatures_multilabel(multilabel_train_data):
    meta_features.helper_functions.clear()
    X, y = multilabel_train_data
    categorical = {i: False for i in range(10)}
    mf = meta_features.calculate_all_metafeatures(
        X, y,  categorical, "Generated", logger=logging.getLogger('TestMeta'))
    assert 52 == len(mf.metafeature_values)


def test_calculate_all_metafeatures_same_results_across_datatypes():
    """
    This test makes sure that numpy and pandas produce the same metafeatures.
    This also is an excuse to fully test anneal dataset, and make sure
    all metafeatures work in this complex dataset
    """
    X, y = fetch_openml(data_id=2, return_X_y=True, as_frame=True)
    categorical = {col: True if X[col].dtype.name == 'category' else False
                   for col in X.columns}
    mf = meta_features.calculate_all_metafeatures(
        X, y, categorical, "2", logger=logging.getLogger('Meta'))
    assert 52 == len(mf.metafeature_values)
    expected = {
        'PCASkewnessFirstPC': 0.41897660337677867,
        'PCAKurtosisFirstPC': -0.677692541156901,
        'PCAFractionOfComponentsFor95PercentVariance': 0.2716049382716049,
        'ClassEntropy': 1.1898338562043977,
        'SkewnessSTD': 7.540418815675546,
        'SkewnessMean': 1.47397188548894,
        'SkewnessMax': 29.916569235579203,
        'SkewnessMin': -29.916569235579203,
        'KurtosisSTD': 153.0563504598898,
        'KurtosisMean': 56.998860939761165,
        'KurtosisMax': 893.0011148272025,
        'KurtosisMin': -3.0,
        'SymbolsSum': 49,
        'SymbolsSTD': 1.3679553264445183,
        'SymbolsMean': 1.8846153846153846,
        'SymbolsMax': 7,
        'SymbolsMin': 1,
        'ClassProbabilitySTD': 0.28282850691819206,
        'ClassProbabilityMean': 0.2,
        'ClassProbabilityMax': 0.7616926503340757,
        'ClassProbabilityMin': 0.008908685968819599,
        'InverseDatasetRatio': 23.63157894736842,
        'DatasetRatio': 0.042316258351893093,
        'RatioNominalToNumerical': 5.333333333333333,
        'RatioNumericalToNominal': 0.1875,
        'NumberOfCategoricalFeatures': 32,
        'NumberOfNumericFeatures': 6,
        'NumberOfMissingValues': 22175.0,
        'NumberOfFeaturesWithMissingValues': 29.0,
        'NumberOfInstancesWithMissingValues': 898.0,
        'NumberOfFeatures': 38.0,
        'NumberOfClasses': 5.0,
        'NumberOfInstances': 898.0,
        'LogInverseDatasetRatio': 3.162583908575814,
        'LogDatasetRatio': -3.162583908575814,
        'PercentageOfMissingValues': 0.6498358926268901,
        'PercentageOfFeaturesWithMissingValues': 0.7631578947368421,
        'PercentageOfInstancesWithMissingValues': 1.0,
        'LogNumberOfFeatures': 3.6375861597263857,
        'LogNumberOfInstances': 6.8001700683022,
    }
    assert {k: mf[k].value for k in expected.keys()} == pytest.approx(expected)

    expected_landmarks = {
        'Landmark1NN': 0.9721601489757914,
        'LandmarkRandomNodeLearner': 0.7616945996275606,
        'LandmarkDecisionNodeLearner':  0.7827932960893855,
        'LandmarkDecisionTree': 0.9899875853507139,
        'LandmarkNaiveBayes': 0.9287150837988827,
        'LandmarkLDA': 0.9610242085661079,
    }
    assert {k: mf[k].value for k in expected_landmarks.keys()} == pytest.approx(
        expected_landmarks, rel=1e-5)

    # Then do numpy!
    X, y = fetch_openml(data_id=2, return_X_y=True, as_frame=False)
    categorical = {i: True if category else False
                   for i, category in enumerate(categorical.values())}
    mf = meta_features.calculate_all_metafeatures(
        X, y, categorical, "2", logger=logging.getLogger('Meta'))
    assert {k: mf[k].value for k in expected.keys()} == pytest.approx(expected)

    # The column-reorder of pandas and numpy array are different after
    # the data preprocessing. So we cannot directly compare, and landmarking is
    # sensible to column order
    expected_landmarks['LandmarkDecisionTree'] = 0.9922098075729361
    assert {k: mf[k].value for k in expected_landmarks.keys()} == pytest.approx(
        expected_landmarks, rel=1e-5)
