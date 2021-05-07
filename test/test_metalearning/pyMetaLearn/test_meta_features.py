import logging
import os
import tempfile
from io import StringIO
import unittest

import pandas as pd

import pytest

import arff
from joblib import Memory
import numpy as np
from sklearn.datasets import make_multilabel_classification

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor
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
    categorical = [True if attribute == 'nominal' else False
                   for attribute in attribute_types]

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
    categorical = [True if attribute == 'nominal' else False
                   for attribute in attribute_types]

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

    DPP = DataPreprocessor(feat_type={
        i: 'categorical' if feat else 'numerical' for i, feat in enumerate(categorical)
    })
    X_transformed = DPP.fit_transform(X)

    number_numerical = np.sum(~np.array(categorical))
    categorical_transformed = [True] * (X_transformed.shape[1] -
                                        number_numerical) + \
                              [False] * number_numerical
    categorical_transformed = categorical_transformed

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


class TestMetaFeatures:
    def test_number_of_instance(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfInstances"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 898
        assert isinstance(mf, MetaFeatureValue)

    def test_number_of_classes(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfClasses"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 5
        assert isinstance(mf, MetaFeatureValue)

    def test_number_of_features(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfFeatures"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 38
        assert isinstance(mf, MetaFeatureValue)

    def test_missing_values(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.helper_functions["MissingValues"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert isinstance(mf.value, pd.DataFrame if hasattr(X, 'iloc') else np.ndarray)
        assert mf.value.shape == X.shape
        # TODO: ASK MATTHIAS FOR THIS CHANGE OF np.sum->np.count_nonzero
        assert 22175 == np.count_nonzero(mf.value)

    def test_number_of_Instances_with_missing_values(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfInstancesWithMissingValues"](
            X, y, logging.getLogger('Meta'), categorical)
        assert mf.value == 898
        assert isinstance(mf, MetaFeatureValue)

    def test_percentage_of_Instances_with_missing_values(self, meta_train_data):
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

    def test_number_of_features_with_missing_values(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfFeaturesWithMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 29
        assert isinstance(mf, MetaFeatureValue)

    def test_percentage_of_features_with_missing_values(self, meta_train_data):
        X, y, categorical = meta_train_data
        meta_features.metafeatures.set_value(
            "NumberOfFeaturesWithMissingValues",
            meta_features.metafeatures["NumberOfFeaturesWithMissingValues"](
                X, y, logging.getLogger('Meta'),  categorical))
        mf = meta_features.metafeatures["PercentageOfFeaturesWithMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert pytest.approx(mf.value) == float(29)/float(38)
        assert isinstance(mf, MetaFeatureValue)

    def test_number_of_missing_values(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 22175
        assert isinstance(mf, MetaFeatureValue)

    def test_percentage_missing_values(self, meta_train_data):
        X, y, categorical = meta_train_data
        meta_features.metafeatures.set_value(
            "NumberOfMissingValues", meta_features.metafeatures["NumberOfMissingValues"](
                X, y, logging.getLogger('Meta'),  categorical))
        mf = meta_features.metafeatures["PercentageOfMissingValues"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert pytest.approx(mf.value) == (float(22175)/float(38*898))
        assert isinstance(mf, MetaFeatureValue)

    def test_number_of_numeric_features(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfNumericFeatures"](
            X, y, logging.getLogger('Meta'), categorical)
        assert mf.value == 6
        assert isinstance(mf, MetaFeatureValue)

    def test_number_of_categorical_features(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["NumberOfCategoricalFeatures"](
            X, y, logging.getLogger('Meta'), categorical)
        assert mf.value == 32
        assert isinstance(mf, MetaFeatureValue)

    def test_ratio_numerical_to_categorical(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["RatioNumericalToNominal"](
            X, y, logging.getLogger('Meta'), categorical)
        assert pytest.approx(mf.value) == (float(6)/float(32))
        assert isinstance(mf, MetaFeatureValue)

    def test_ratio_categorical_to_numerical(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["RatioNominalToNumerical"](
            X, y, logging.getLogger('Meta'), categorical)
        assert pytest.approx(mf.value) == (float(32)/float(6))
        assert isinstance(mf, MetaFeatureValue)

    def test_dataset_ratio(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["DatasetRatio"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert pytest.approx(mf.value) == (float(38)/float(898))
        assert isinstance(mf, MetaFeatureValue)

    def test_inverse_dataset_ratio(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["InverseDatasetRatio"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert pytest.approx(mf.value) == (float(898)/float(38))
        assert isinstance(mf, MetaFeatureValue)

    def test_class_occurences(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.helper_functions["ClassOccurences"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == {0.0: 8.0, 1.0: 99.0, 2.0: 684.0, 4.0: 67.0, 5.0: 40.0}

    def test_class_probability_min(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["ClassProbabilityMin"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert pytest.approx(mf.value) == (float(8)/float(898))
        assert isinstance(mf, MetaFeatureValue)

    def test_class_probability_max(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["ClassProbabilityMax"](
            X, y, logging.getLogger('Meta'),  categorical)
        assert pytest.approx(mf.value) == (float(684)/float(898))
        assert isinstance(mf, MetaFeatureValue)

    def test_class_probability_mean(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["ClassProbabilityMean"](
            X, y, logging.getLogger('Meta'),  categorical)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_mean = (classes / float(898)).mean()
        assert pytest.approx(mf.value) == prob_mean
        assert isinstance(mf, MetaFeatureValue)

    def test_class_probability_std(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["ClassProbabilitySTD"](
            X, y, logging.getLogger('Meta'),  categorical)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_std = (classes / float(898)).std()
        assert pytest.approx(mf.value) == prob_std
        assert isinstance(mf, MetaFeatureValue)

    def test_num_symbols(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.helper_functions["NumSymbols"](
            X, y, logging.getLogger('Meta'),  categorical)
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, 0,
                            1, 1, 1, 0, 1, 1, 0, 3, 1, 0, 0, 0, 2, 2, 3, 2]
        assert mf.value == symbol_frequency

    def test_symbols_min(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["SymbolsMin"](X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 1

    def test_symbols_max(self, meta_train_data):
        X, y, categorical = meta_train_data
        # this is attribute steel
        mf = meta_features.metafeatures["SymbolsMax"](X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 7

    def test_symbols_mean(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["SymbolsMean"](
            X, y, logging.getLogger('Meta'),  categorical)
        # Empty looking spaces denote empty attributes
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1,  #
                            1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
        assert pytest.approx(mf.value) == np.mean(symbol_frequency)

    def test_symbols_std(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["SymbolsSTD"](X, y, logging.getLogger('Meta'),  categorical)
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1,  #
                            1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
        assert pytest.approx(mf.value) == np.std(symbol_frequency)

    def test_symbols_sum(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["SymbolsSum"](X, y, logging.getLogger('Meta'),  categorical)
        assert mf.value == 49

    def test_class_entropy(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.metafeatures["ClassEntropy"](
            X, y, logging.getLogger('Meta'),  categorical)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        classes = classes / sum(classes)
        entropy = -np.sum([c * np.log2(c) for c in classes])

        assert pytest.approx(mf.value) == entropy

    def test_calculate_all_metafeatures(self, meta_train_data):
        X, y, categorical = meta_train_data
        mf = meta_features.calculate_all_metafeatures(
            X, y, categorical, "2", logger=logging.getLogger('Meta'))
        assert 52 == len(mf.metafeature_values)
        assert mf.metafeature_values['NumberOfCategoricalFeatures'].value == 32
        sio = StringIO()
        mf.dump(sio)


class TestMetaFeaturesTransformed:
    def test_kurtosisses(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        mf = meta_features.helper_functions["Kurtosisses"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)
        assert 6 == len(mf.value)

    def test_kurtosis_min(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["KurtosisMin"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_kurtosis_max(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["KurtosisMax"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_kurtosis_mean(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["KurtosisMean"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_kurtosis_std(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["KurtosisSTD"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_skewnesses(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        mf = meta_features.helper_functions["Skewnesses"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)
        assert 6 == len(mf.value)

    def test_skewness_min(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["SkewnessMin"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_skewness_max(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["SkewnessMax"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_skewness_mean(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["SkewnessMean"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_skewness_std(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["SkewnessSTD"](
            X_transformed, y, logging.getLogger('Meta'), categorical_transformed)

    def test_landmark_lda(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["LandmarkLDA"](X_transformed, y, logging.getLogger('Meta'))

    def test_landmark_naive_bayes(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["LandmarkNaiveBayes"](
            X_transformed, y, logging.getLogger('Meta'))

    def test_landmark_decision_tree(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["LandmarkDecisionTree"](
            X_transformed, y, logging.getLogger('Meta'))

    def test_decision_node(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["LandmarkDecisionNodeLearner"](
            X_transformed, y, logging.getLogger('Meta'))

    def test_random_node(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["LandmarkRandomNodeLearner"](
            X_transformed, y, logging.getLogger('Meta'))

    @unittest.skip("Currently not implemented!")
    def test_worst_node(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["LandmarkWorstNodeLearner"](
            X_transformed, y, logging.getLogger('Meta'))

    def test_1NN(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        # TODO: somehow compute the expected output?
        meta_features.metafeatures["Landmark1NN"](X_transformed, y, logging.getLogger('Meta'))

    def test_pca(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        meta_features.helper_functions["PCA"](X_transformed, y, logging.getLogger('Meta'))

    def test_pca_95percent(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        mf = meta_features.metafeatures["PCAFractionOfComponentsFor95PercentVariance"](
            X_transformed, y, logging.getLogger('Meta'))
        assert pytest.approx(0.2716049382716049) == mf.value

    def test_pca_kurtosis_first_pc(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        mf = meta_features.metafeatures["PCAKurtosisFirstPC"](
            X_transformed, y, logging.getLogger('Meta'))
        assert pytest.approx(-0.702850) != mf.value

    def test_pca_skewness_first_pc(self, meta_train_data_transformed):
        X_transformed, y, categorical_transformed = meta_train_data_transformed
        mf = meta_features.metafeatures["PCASkewnessFirstPC"](
            X_transformed, y, logging.getLogger('Meta'))
        assert pytest.approx(0.051210) != mf.value


class TestMetaFeaturesMultiLabel:
    def test_class_occurences_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.helper_functions["ClassOccurences"](X, y, logging.getLogger('Meta'))
        assert mf.value == [{0: 16.0, 1: 84.0},
                            {0: 8.0, 1: 92.0},
                            {0: 68.0, 1: 32.0},
                            {0: 15.0, 1: 85.0},
                            {0: 28.0, 1: 72.0}]

    def test_class_probability_min_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        meta_features.helper_functions.set_value(
            "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
                X, y, logging.getLogger('Meta')))
        mf = meta_features.metafeatures["ClassProbabilityMin"](X, y, logging.getLogger('Meta'))
        assert pytest.approx(mf.value) == (float(8) / float(100))
        assert isinstance(mf, MetaFeatureValue)

    def test_class_probability_max_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        meta_features.helper_functions.set_value(
            "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
                X, y, logging.getLogger('Meta')))
        mf = meta_features.metafeatures["ClassProbabilityMax"](X, y, logging.getLogger('Meta'))
        assert pytest.approx(mf.value) == (float(92) / float(100))
        assert isinstance(mf, MetaFeatureValue)

    def test_class_probability_mean_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        meta_features.helper_functions.set_value(
            "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
                X, y, logging.getLogger('Meta')))
        mf = meta_features.metafeatures["ClassProbabilityMean"](X, y, logging.getLogger('Meta'))
        classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
        probas = np.mean([np.mean(np.array(cls_)) / 100 for cls_ in classes])
        assert mf.value == pytest.approx(probas)
        assert isinstance(mf, MetaFeatureValue)

    def test_number_of_classes_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["NumberOfClasses"](X, y, logging.getLogger('Meta'))
        assert mf.value == 2
        assert isinstance(mf, MetaFeatureValue)

    def test_class_probability_std_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        meta_features.helper_functions.set_value(
            "ClassOccurences", meta_features.helper_functions["ClassOccurences"](
                X, y, logging.getLogger('Meta')))
        mf = meta_features.metafeatures["ClassProbabilitySTD"](X, y, logging.getLogger('Meta'))
        classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
        probas = np.mean([np.std(np.array(cls_) / 100.) for cls_ in classes])
        assert pytest.approx(mf.value) == probas
        assert isinstance(mf, MetaFeatureValue)

    def test_class_entropy_multilabel(self, multilabel_train_data):
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

    def test_landmark_lda_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["LandmarkLDA"](X, y, logging.getLogger('Meta'))
        assert np.isfinite(mf.value)

    def test_landmark_naive_bayes_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["LandmarkNaiveBayes"](X, y, logging.getLogger('Meta'))
        assert np.isfinite(mf.value)

    def test_landmark_decision_tree_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["LandmarkDecisionTree"](X, y, logging.getLogger('Meta'))
        assert np.isfinite(mf.value)

    def test_landmark_decision_node_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["LandmarkDecisionNodeLearner"](
            X, y, logging.getLogger('Meta'))
        assert np.isfinite(mf.value)

    def test_landmark_random_node_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["LandmarkRandomNodeLearner"](
            X, y, logging.getLogger('Meta'))
        assert np.isfinite(mf.value)

    def test_1NN_multilabel(self, multilabel_train_data):
        X, y = multilabel_train_data
        mf = meta_features.metafeatures["Landmark1NN"](X, y, logging.getLogger('TestMeta'))
        assert np.isfinite(mf.value)

    def test_calculate_all_metafeatures_multilabel(self, multilabel_train_data):
        meta_features.helper_functions.clear()
        X, y = multilabel_train_data
        categorical = [False] * 10
        mf = meta_features.calculate_all_metafeatures(
            X, y,  categorical, "Generated", logger=logging.getLogger('TestMeta'))
        assert 52 == len(mf.metafeature_values)
        sio = StringIO()
        mf.dump(sio)
