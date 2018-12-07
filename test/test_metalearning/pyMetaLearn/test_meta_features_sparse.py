from io import StringIO
import os
import sys
import unittest

# Make the super class importable
sys.path.append(os.path.dirname(__file__))

import arff
import numpy as np
from scipy import sparse
from sklearn.preprocessing.imputation import Imputer

from autosklearn.pipeline.implementations.OneHotEncoder import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import autosklearn.metalearning.metafeatures.metafeatures as meta_features
import test_meta_features


class SparseMetaFeaturesTest(test_meta_features.MetaFeaturesTest,
                             unittest.TestCase):
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
        X = data[:, :-1]
        y = data[:, -1].reshape((-1,))

        # First, swap NaNs and zeros, because when converting an encoded
        # dense matrix to sparse, the values which are encoded to zero are lost
        X_sparse = X.copy()
        NaNs = ~np.isfinite(X_sparse)
        X_sparse[NaNs] = 0
        X_sparse = sparse.csr_matrix(X_sparse)

        ohe = OneHotEncoder(self.categorical)
        X_transformed = X_sparse.copy()
        X_transformed = ohe.fit_transform(X_transformed)
        imp = Imputer(copy=False)
        X_transformed = imp.fit_transform(X_transformed)
        standard_scaler = StandardScaler(with_mean=False)
        X_transformed = standard_scaler.fit_transform(X_transformed)

        # Transform the array which indicates the categorical metafeatures
        number_numerical = np.sum(~np.array(self.categorical))
        categorical_transformed = [True] * (X_transformed.shape[1] -
                                            number_numerical) + \
                                  [False] * number_numerical
        self.categorical_transformed = categorical_transformed

        self.X = X_sparse
        self.X_transformed = X_transformed
        self.y = y
        self.mf = meta_features.metafeatures
        self.helpers = meta_features.helper_functions

        # Precompute some helper functions
        self.helpers.set_value("PCA", self.helpers["PCA"]
            (self.X_transformed, self.y))
        self.helpers.set_value("MissingValues", self.helpers[
            "MissingValues"](self.X, self.y, self.categorical))
        self.mf.set_value("NumberOfMissingValues",
            self.mf["NumberOfMissingValues"](self.X, self.y, self.categorical))
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

    def test_missing_values(self):
        mf = self.helpers["MissingValues"](self.X, self.y, self.categorical)
        self.assertTrue(sparse.issparse(mf.value))
        self.assertEqual(mf.value.shape, self.X.shape)
        self.assertEqual(mf.value.dtype, np.bool)
        self.assertEqual(0, np.sum(mf.value.data))

    def test_number_of_missing_values(self):
        mf = self.mf["NumberOfMissingValues"](self.X, self.y, self.categorical)
        self.assertEqual(0, mf.value)

    def test_percentage_missing_values(self):
        mf = self.mf["PercentageOfMissingValues"](self.X, self.y, self.categorical)
        self.assertEqual(0, mf.value)

    def test_number_of_Instances_with_missing_values(self):
        mf = self.mf["NumberOfInstancesWithMissingValues"](
            self.X, self.y, self.categorical)
        self.assertEqual(0, mf.value)

    def test_percentage_of_Instances_with_missing_values(self):
        self.mf.set_value("NumberOfInstancesWithMissingValues",
                          self.mf["NumberOfInstancesWithMissingValues"](
                              self.X, self.y, self.categorical))
        mf = self.mf["PercentageOfInstancesWithMissingValues"](self.X, self.y,
                                                               self.categorical)
        self.assertAlmostEqual(0, mf.value)

    def test_number_of_features_with_missing_values(self):
        mf = self.mf["NumberOfFeaturesWithMissingValues"](self.X, self.y,
                                                          self.categorical)
        self.assertEqual(0, mf.value)

    def test_percentage_of_features_with_missing_values(self):
        self.mf.set_value("NumberOfFeaturesWithMissingValues",
                          self.mf["NumberOfFeaturesWithMissingValues"](
                              self.X, self.y, self.categorical))
        mf = self.mf["PercentageOfFeaturesWithMissingValues"](self.X, self.y,
                                                              self.categorical)
        self.assertAlmostEqual(0, mf.value)

    def test_num_symbols(self):
        mf = self.helpers["NumSymbols"](self.X, self.y, self.categorical)

        symbol_frequency = [2, 0, 6, 0, 1, 3, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 2, 2]
        self.assertEqual(mf.value, symbol_frequency)

    def test_symbols_max(self):
        # this is attribute steel
        mf = self.mf["SymbolsMax"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 6)

    def test_symbols_mean(self):
        mf = self.mf["SymbolsMean"](self.X, self.y, self.categorical)
        # Empty looking spaces denote empty attributes
        symbol_frequency = [2, 6, 1, 3, 3, 1, 1, 2, 1, 1, 2, 2]
        self.assertAlmostEqual(mf.value, np.mean(symbol_frequency))

    def test_symbols_std(self):
        mf = self.mf["SymbolsSTD"](self.X, self.y, self.categorical)
        symbol_frequency = [2, 6, 1, 3, 3, 1, 1, 2, 1, 1, 2, 2]
        self.assertAlmostEqual(mf.value, np.std(symbol_frequency))

    def test_symbols_sum(self):
        mf = self.mf["SymbolsSum"](self.X, self.y, self.categorical)
        self.assertEqual(mf.value, 25)

    def test_skewnesses(self):
        fixture = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                   -0.6969708499033568, 0.626346013011263,
                   0.3809987596624038, 1.4762248835141034,
                   0.07687661087633726, 0.36889797830360116]
        mf = self.helpers["Skewnesses"](self.X_transformed, self.y)
        print(mf.value)
        print(fixture)
        np.testing.assert_allclose(mf.value, fixture)

    def test_kurtosisses(self):
        fixture = [-3.0, -3.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0,
                   -3.0, -2.0, -3.0, -2.0, -3.0, -3.0, -2.0, -3.0,
                   -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -3.0,
                   -3.0, -3.0, -1.1005836114255765,
                   -1.1786325509475712, -1.2387998382327912,
                   1.393438264413704, -0.9768209837948336,
                   -1.7937072296512782]
        mf = self.helpers["Kurtosisses"](self.X_transformed, self.y)
        np.testing.assert_allclose(mf.value, fixture)

    def test_pca_95percent(self):
        mf = self.mf["PCAFractionOfComponentsFor95PercentVariance"](
            self.X_transformed, self.y)
        self.assertAlmostEqual(0.7741935483870968, mf.value)

    def test_pca_kurtosis_first_pc(self):
        mf = self.mf["PCAKurtosisFirstPC"](self.X_transformed, self.y)
        self.assertAlmostEqual(-0.15444516166802469, mf.value)

    def test_pca_skewness_first_pc(self):
        mf = self.mf["PCASkewnessFirstPC"](self.X_transformed, self.y)
        self.assertAlmostEqual(0.026514792083623905, mf.value)

    def test_calculate_all_metafeatures(self):
        mf = meta_features.calculate_all_metafeatures(
            self.X, self.y, self.categorical, "2")
        self.assertEqual(52, len(mf.metafeature_values))
        sio = StringIO()
        mf.dump(sio)
