import os
import tempfile
from io import StringIO
from unittest import TestCase
import unittest

import arff
import numpy as np
import scipy.sparse
from sklearn.preprocessing.imputation import Imputer
from sklearn.datasets import make_multilabel_classification
from sklearn.externals.joblib import Memory

from autosklearn.pipeline.implementations.OneHotEncoder import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from autosklearn.metalearning.metafeatures.metafeature import MetaFeatureValue
import autosklearn.metalearning.metafeatures.metafeatures as meta_features


class MetaFeaturesTest(TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.cwd = os.getcwd()
        tests_dir = __file__
        os.chdir(os.path.dirname(tests_dir))

        decoder = arff.ArffDecoder()
        with open(os.path.join("datasets", "dataset.arff")) as fh:
            dataset = decoder.decode(fh, encode_nominal=True)

        # -1 because the last attribute is the class
        self.attribute_types = [
            'numeric' if type(type_) != list else 'nominal'
            for name, type_ in dataset['attributes'][:-1]]
        self.categorical = [True if attribute == 'nominal' else False
                            for attribute in self.attribute_types]

        data = np.array(dataset['data'], dtype=np.float64)
        X = data[:,:-1]
        y = data[:,-1].reshape((-1,))

        ohe = OneHotEncoder(self.categorical)
        X_transformed = ohe.fit_transform(X)
        imp = Imputer(copy=False)
        X_transformed = imp.fit_transform(X_transformed)
        center = not scipy.sparse.isspmatrix((X_transformed))
        standard_scaler = StandardScaler(with_mean=center)
        X_transformed = standard_scaler.fit_transform(X_transformed)
        X_transformed = X_transformed.todense()

        # Transform the array which indicates the categorical metafeatures
        number_numerical = np.sum(~np.array(self.categorical))
        categorical_transformed = [True] * (X_transformed.shape[1] -
                                            number_numerical) + \
                                  [False] * number_numerical
        self.categorical_transformed = categorical_transformed

        self.X = X
        self.X_transformed = X_transformed
        self.y = y
        self.mf = meta_features.metafeatures
        self.helpers = meta_features.helper_functions

        # Precompute some helper functions
        self.helpers.set_value("PCA", self.helpers["PCA"]
            (self.X_transformed, self.y))
        self.helpers.set_value("MissingValues", self.helpers[
            "MissingValues"](self.X, self.y, self.categorical))
        self.helpers.set_value("NumSymbols", self.helpers["NumSymbols"](
            self.X, self.y, self.categorical))
        self.helpers.set_value("ClassOccurences",
                               self.helpers["ClassOccurences"](self.X, self.y))
        self.helpers.set_value("Skewnesses",
            self.helpers["Skewnesses"](self.X_transformed, self.y,
                                       self.categorical_transformed))
        self.helpers.set_value("Kurtosisses",
            self.helpers["Kurtosisses"](self.X_transformed, self.y,
                                        self.categorical_transformed))

    def tearDown(self):
        os.chdir(self.cwd)

    def get_multilabel(self):
        cache = Memory(cachedir=tempfile.gettempdir())
        cached_func = cache.cache(make_multilabel_classification)
        return cached_func(
            n_samples=100,
            n_features=10,
            n_classes=5,
            n_labels=5,
            return_indicator=True,
            random_state=1
        )

    def test_number_of_instance(self):
        mf = self.mf["NumberOfInstances"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 898)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_classes(self):
        mf = self.mf["NumberOfClasses"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 5)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_classes_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["NumberOfClasses"](X, y)
        self.assertEqual(mf.value, 2)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_features(self):
        mf = self.mf["NumberOfFeatures"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 38)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_missing_values(self):
        mf = self.helpers["MissingValues"](self.X, self.y, self.categorical)
        self.assertIsInstance(mf.value, np.ndarray)
        self.assertEqual(mf.value.shape, self.X.shape)
        self.assertEqual(22175, np.sum(mf.value))

    def test_number_of_Instances_with_missing_values(self):
        mf = self.mf["NumberOfInstancesWithMissingValues"](self.X, self.y,
                                                           self.categorical)
        self.assertEqual(mf.value, 898)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_percentage_of_Instances_with_missing_values(self):
        self.mf.set_value("NumberOfInstancesWithMissingValues",
            self.mf["NumberOfInstancesWithMissingValues"](self.X, self.y,
                                                               self.categorical))
        mf = self.mf["PercentageOfInstancesWithMissingValues"](self.X, self.y,
                                                               self.categorical)
        self.assertAlmostEqual(mf.value, 1.0)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_features_with_missing_values(self):
        mf = self.mf["NumberOfFeaturesWithMissingValues"](self.X, self.y,
                                                          self.categorical)
        self.assertEqual(mf.value, 29)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_percentage_of_features_with_missing_values(self):
        self.mf.set_value("NumberOfFeaturesWithMissingValues",
            self.mf["NumberOfFeaturesWithMissingValues"](self.X, self.y,
                                                         self.categorical))
        mf = self.mf["PercentageOfFeaturesWithMissingValues"](self.X, self.y,
                                                              self.categorical)
        self.assertAlmostEqual(mf.value, float(29)/float(38))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_missing_values(self):
        mf = self.mf["NumberOfMissingValues"](self.X, self.y,
                                                 self.categorical)
        self.assertEqual(mf.value, 22175)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_percentage_missing_values(self):
        self.mf.set_value("NumberOfMissingValues",
                          self.mf["NumberOfMissingValues"](self.X, self.y,
                                                           self.categorical))
        mf = self.mf["PercentageOfMissingValues"](self.X, self.y,
                                                  self.categorical)
        self.assertAlmostEqual(mf.value, float(22175)/float((38*898)))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_numeric_features(self):
        mf = self.mf["NumberOfNumericFeatures"](self.X, self.y,
                                                self.categorical)
        self.assertEqual(mf.value, 6)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_number_of_categorical_features(self):
        mf = self.mf["NumberOfCategoricalFeatures"](self.X, self.y,
                                                    self.categorical)
        self.assertEqual(mf.value, 32)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_ratio_numerical_to_categorical(self):
        mf = self.mf["RatioNumericalToNominal"](self.X, self.y,
                                                self.categorical)
        self.assertAlmostEqual(mf.value, float(6)/float(32))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_ratio_categorical_to_numerical(self):
        mf = self.mf["RatioNominalToNumerical"](self.X, self.y,
                                                self.categorical)
        self.assertAlmostEqual(mf.value, float(32)/float(6))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_dataset_ratio(self):
        mf = self.mf["DatasetRatio"](self.X, self.y, self.categorical)
        self.assertAlmostEqual(mf.value, float(38)/float(898))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_inverse_dataset_ratio(self):
        mf = self.mf["InverseDatasetRatio"](self.X, self.y, self.categorical)
        self.assertAlmostEqual(mf.value, float(898)/float(38))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_occurences(self):
        mf = self.helpers["ClassOccurences"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value,
                         {0.0: 8.0, 1.0: 99.0, 2.0: 684.0, 4.0: 67.0, 5.0: 40.0})

    def test_class_occurences_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.helpers["ClassOccurences"](X, y)
        self.assertEqual(mf.value,
                         [{0: 16.0, 1: 84.0},
                          {0: 8.0, 1: 92.0},
                          {0: 68.0, 1: 32.0},
                          {0: 15.0, 1: 85.0},
                          {0: 28.0, 1: 72.0}])

    def test_class_probability_min(self):
        mf = self.mf["ClassProbabilityMin"](self.X, self.y, self.categorical)
        self.assertAlmostEqual(mf.value, float(8)/float(898))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_min_multilabel(self):
        X, y = self.get_multilabel()
        self.helpers.set_value("ClassOccurences",
                               self.helpers["ClassOccurences"](X, y))
        mf = self.mf["ClassProbabilityMin"](X, y)
        self.assertAlmostEqual(mf.value, float(8) / float(100))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_max(self):
        mf = self.mf["ClassProbabilityMax"](self.X, self.y, self.categorical)
        self.assertAlmostEqual(mf.value, float(684)/float(898))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_max_multilabel(self):
        X, y = self.get_multilabel()
        self.helpers.set_value("ClassOccurences",
                               self.helpers["ClassOccurences"](X, y))
        mf = self.mf["ClassProbabilityMax"](X, y)
        self.assertAlmostEqual(mf.value, float(92) / float(100))
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_mean(self):
        mf = self.mf["ClassProbabilityMean"](self.X, self.y, self.categorical)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_mean = (classes / float(898)).mean()
        self.assertAlmostEqual(mf.value, prob_mean)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_mean_multilabel(self):
        X, y = self.get_multilabel()
        self.helpers.set_value("ClassOccurences",
                               self.helpers["ClassOccurences"](X, y))
        mf = self.mf["ClassProbabilityMean"](X, y)
        classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
        probas = np.mean([np.mean(np.array(cls_)) / 100 for cls_ in classes])
        self.assertAlmostEqual(mf.value, probas)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_std(self):
        mf = self.mf["ClassProbabilitySTD"](self.X, self.y, self.categorical)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        prob_std = (classes / float(898)).std()
        self.assertAlmostEqual(mf.value, prob_std)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_class_probability_std_multilabel(self):
        X, y = self.get_multilabel()
        self.helpers.set_value("ClassOccurences",
                               self.helpers["ClassOccurences"](X, y))
        mf = self.mf["ClassProbabilitySTD"](X, y)
        classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
        probas = np.mean([np.std(np.array(cls_) / 100.) for cls_ in classes])
        self.assertAlmostEqual(mf.value, probas)
        self.assertIsInstance(mf, MetaFeatureValue)

    def test_num_symbols(self):
        mf = self.helpers["NumSymbols"](self.X, self.y, self.categorical)
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, 0,
                            1, 1, 1, 0, 1, 1, 0, 3, 1, 0, 0, 0, 2, 2, 3, 2]
        self.assertEqual(mf.value, symbol_frequency)

    def test_symbols_min(self):
        mf = self.mf["SymbolsMin"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 1)

    def test_symbols_max(self):
        # this is attribute steel
        mf = self.mf["SymbolsMax"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 7)

    def test_symbols_mean(self):
        mf = self.mf["SymbolsMean"](self.X, self.y, self.categorical)
        # Empty looking spaces denote empty attributes
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, #
                            1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
        self.assertAlmostEqual(mf.value, np.mean(symbol_frequency))

    def test_symbols_std(self):
        mf = self.mf["SymbolsSTD"](self.X, self.y, self.categorical)
        symbol_frequency = [2, 1, 7, 1, 2, 4, 1, 1, 4, 2, 1, 1, 1, 2, 1, #
                            1, 1, 1,   1, 1,    3, 1,           2, 2, 3, 2]
        self.assertAlmostEqual(mf.value, np.std(symbol_frequency))

    def test_symbols_sum(self):
        mf = self.mf["SymbolsSum"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 49)

    def test_kurtosisses(self):
        mf = self.helpers["Kurtosisses"](self.X_transformed, self.y,
                                         self.categorical_transformed)
        self.assertEqual(6, len(mf.value))

    def test_kurtosis_min(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["KurtosisMin"](self.X_transformed, self.y,
                                    self.categorical_transformed)

    def test_kurtosis_max(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["KurtosisMax"](self.X_transformed, self.y,
                                    self.categorical_transformed)

    def test_kurtosis_mean(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["KurtosisMean"](self.X_transformed, self.y,
                                     self.categorical_transformed)

    def test_kurtosis_std(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["KurtosisSTD"](self.X_transformed, self.y,
                                    self.categorical_transformed)

    def test_skewnesses(self):
        mf = self.helpers["Skewnesses"](self.X_transformed, self.y,
                                        self.categorical_transformed)
        self.assertEqual(6, len(mf.value))

    def test_skewness_min(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["SkewnessMin"](self.X_transformed, self.y,
                                    self.categorical_transformed)

    def test_skewness_max(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["SkewnessMax"](self.X_transformed, self.y,
                                    self.categorical_transformed)

    def test_skewness_mean(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["SkewnessMean"](self.X_transformed, self.y,
                                     self.categorical_transformed)

    def test_skewness_std(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["SkewnessSTD"](self.X_transformed, self.y,
                                    self.categorical_transformed)

    def test_class_entropy(self):
        mf = self.mf["ClassEntropy"](self.X, self.y, self.categorical)
        classes = np.array((8, 99, 684, 67, 40), dtype=np.float64)
        classes = classes / sum(classes)
        entropy = -np.sum([c * np.log2(c) for c in classes])

        self.assertAlmostEqual(mf.value, entropy)

    def test_class_entropy_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["ClassEntropy"](X, y)

        classes = [(16, 84), (8, 92), (68, 32), (15, 85), (28, 72)]
        entropies = []
        for cls in classes:
            cls = np.array(cls, dtype=np.float32)
            cls = cls / sum(cls)
            entropy = -np.sum([c * np.log2(c) for c in cls])
            entropies.append(entropy)

        self.assertAlmostEqual(mf.value, np.mean(entropies))

    def test_landmark_lda(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["LandmarkLDA"](self.X_transformed, self.y)

    def test_landmark_lda_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["LandmarkLDA"](X, y)
        self.assertTrue(np.isfinite(mf.value))

    def test_landmark_naive_bayes(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["LandmarkNaiveBayes"](self.X_transformed, self.y)

    def test_landmark_naive_bayes_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["LandmarkNaiveBayes"](X, y)
        self.assertTrue(np.isfinite(mf.value))

    def test_landmark_decision_tree(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["LandmarkDecisionTree"](self.X_transformed, self.y)

    def test_landmark_decision_tree_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["LandmarkDecisionTree"](X, y)
        self.assertTrue(np.isfinite(mf.value))

    def test_decision_node(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["LandmarkDecisionNodeLearner"](self.X_transformed, self.y)

    def test_landmark_decision_node_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["LandmarkDecisionNodeLearner"](X, y)
        self.assertTrue(np.isfinite(mf.value))

    def test_random_node(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["LandmarkRandomNodeLearner"](self.X_transformed, self.y)

    def test_landmark_random_node_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["LandmarkRandomNodeLearner"](X, y)
        self.assertTrue(np.isfinite(mf.value))

    @unittest.skip("Currently not implemented!")
    def test_worst_node(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["LandmarkWorstNodeLearner"](self.X_transformed, self.y)

    def test_1NN(self):
        # TODO: somehow compute the expected output?
        mf = self.mf["Landmark1NN"](self.X_transformed, self.y)

    def test_1NN_multilabel(self):
        X, y = self.get_multilabel()
        mf = self.mf["Landmark1NN"](X, y)
        self.assertTrue(np.isfinite(mf.value))

    def test_pca(self):
        hf = self.helpers["PCA"](self.X_transformed, self.y)

    def test_pca_95percent(self):
        mf = self.mf["PCAFractionOfComponentsFor95PercentVariance"](self.X_transformed, self.y)
        self.assertAlmostEqual(0.44047619047619047, mf.value)

    def test_pca_kurtosis_first_pc(self):
        mf = self.mf["PCAKurtosisFirstPC"](self.X_transformed, self.y)
        self.assertNotAlmostEqual(-0.702850, mf.value)

    def test_pca_skewness_first_pc(self):
        mf = self.mf["PCASkewnessFirstPC"](self.X_transformed, self.y)
        self.assertNotAlmostEqual(0.051210, mf.value)

    def test_calculate_all_metafeatures(self):
        mf = meta_features.calculate_all_metafeatures(
            self.X, self.y, self.categorical, "2")
        self.assertEqual(52, len(mf.metafeature_values))
        self.assertEqual(mf.metafeature_values[
                             'NumberOfCategoricalFeatures'].value, 32)
        sio = StringIO()
        mf.dump(sio)

    def test_calculate_all_metafeatures_multilabel(self):
        self.helpers.clear()
        X, y = self.get_multilabel()
        categorical = [False] * 10
        mf = meta_features.calculate_all_metafeatures(
            X, y, categorical, "Generated")
        self.assertEqual(52, len(mf.metafeature_values))
        sio = StringIO()
        mf.dump(sio)


if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestMetaFeatures)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    t = unittest.TestLoader().loadTestsFromName(
        "pyMetaLearn.metafeatures.test_meta_features.TestMetaFeatures"
        ".test_calculate_all_metafeatures")
    unittest.TextTestRunner(verbosity=2).run(t)
