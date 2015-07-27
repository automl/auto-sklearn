from StringIO import StringIO
import itertools
import os
import sys

# Make the super class importable
sys.path.append(os.path.dirname(__file__))

import arff
import numpy as np
from scipy import sparse
from sklearn.preprocessing.imputation import Imputer

from ParamSklearn.implementations.OneHotEncoder import OneHotEncoder
from ParamSklearn.implementations.StandardScaler import StandardScaler

import autosklearn.metalearning.metafeatures.metafeatures as meta_features
import test_meta_features


class SparseMetaFeaturesTest(test_meta_features.MetaFeaturesTest):
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
        zeros = X_sparse == 0
        X_sparse[NaNs] = 0
        X_sparse = sparse.csr_matrix(X_sparse)
        counter = 0
        for i, j in itertools.product(range(X.shape[0]), range(X.shape[1])):
            # Set values which were originally zero back to zero
            if zeros[i, j]:
                X_sparse[i, j] = 0
            # Set one fourth of the original NaNs back to NaN
            if i % 2 == 0 and j % 2 == 0 and NaNs[i, j]:
                X_sparse[i, j] = np.NaN
                counter += 1

        ohe = OneHotEncoder(self.categorical)
        X_transformed = X_sparse.copy()
        X_transformed = ohe.fit_transform(X_transformed)
        imp = Imputer(copy=False)
        X_transformed = imp.fit_transform(X_transformed)
        standard_scaler = StandardScaler()
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
        self.assertEqual(5319, np.sum(mf.value.data))

    def test_number_of_missing_values(self):
        mf = self.mf["NumberOfMissingValues"](self.X, self.y, self.categorical)
        self.assertEqual(5319, mf.value)

    def test_percentage_missing_values(self):
        mf = self.mf["PercentageOfMissingValues"](self.X, self.y, self.categorical)
        self.assertEqual(float(5319)/float(898*38), mf.value)

    def test_number_of_Instances_with_missing_values(self):
        mf = self.mf["NumberOfInstancesWithMissingValues"](
            self.X, self.y, self.categorical)
        self.assertEqual(898/2, mf.value)

    def test_percentage_of_Instances_with_missing_values(self):
        self.mf.set_value("NumberOfInstancesWithMissingValues",
                          self.mf["NumberOfInstancesWithMissingValues"](
                              self.X, self.y, self.categorical))
        mf = self.mf["PercentageOfInstancesWithMissingValues"](self.X, self.y,
                                                               self.categorical)
        self.assertAlmostEqual(0.5, mf.value)

    def test_number_of_features_with_missing_values(self):
        mf = self.mf["NumberOfFeaturesWithMissingValues"](self.X, self.y,
                                                          self.categorical)
        self.assertEqual(14, mf.value)

    def test_percentage_of_features_with_missing_values(self):
        self.mf.set_value("NumberOfFeaturesWithMissingValues",
                          self.mf["NumberOfFeaturesWithMissingValues"](
                              self.X, self.y, self.categorical))
        mf = self.mf["PercentageOfFeaturesWithMissingValues"](self.X, self.y,
                                                              self.categorical)
        self.assertAlmostEqual(float(14)/float(38), mf.value)

    def test_skewnesses(self):
        mf = self.helpers["Skewnesses"](self.X_transformed, self.y)
        self.assertEqual(str([0.0, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan,
                              np.nan, 0.0, 0.0, -2.1717380859814943, 0.0,
                              np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              -1.3406501827593897, 0.0, 0.0, 0.0, 0.0, np.nan,
                              0.0, 0.0, 6.8652365656298615, 0.0, 0.0, 0.0, 0.0,
                              8.456571731037167, 0.0, 0.0, 1.1517139198634414,
                              0.0, 0.7689648524911159, 0.0, 0.0, 0.0, 0.0,
                              7.856842927127302, 0.0, 0.0, 0.0,
                              3.563940300169727, np.nan, 10.47659050196491,
                              np.nan, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0,
                              -0.6969708499033568, 0.626346013011263,
                              0.38099875966240376, 1.4762248835141034,
                              0.07687661087633728, 0.36889797830360116]),
                         str(mf.value))

    def test_kurtosisses(self):
        mf = self.helpers["Kurtosisses"](self.X_transformed, self.y)
        self.assertEqual(str([-3.0, np.nan, np.nan, -3.0, np.nan, np.nan, np.nan, np.nan, -3.0, -3.0,
                              2.716446314102563, -3.0, np.nan, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
                              -0.20265708746721423, -3.0, -3.0, -3.0, -3.0, np.nan, -3.0, -3.0,
                              45.131473102061314, -3.0, -3.0, -3.0, -3.0, 69.51360544217692,
                              -3.0, -3.0, -0.6735550467927869, -3.0, -1.4086930556333161, -3.0,
                              -3.0, -3.0, -3.0, 59.729980781550324, -3.0, -3.0, -3.0,
                              10.701670463173883, np.nan, 107.75894854586127, np.nan, -3.0, -3.0, -3.0,
                              -3.0, np.nan, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, np.nan,
                              -3.0, -3.0, -1.100583611425576, -1.1786325509475737,
                              -1.2387998382327916, 1.3934382644137013, -0.9768209837948341,
                              -1.7937072296512782]), str(mf.value))

    def test_pca_95percent(self):
        mf = self.mf["PCAFractionOfComponentsFor95PercentVariance"](
            self.X_transformed, self.y)
        self.assertAlmostEqual(0.3611111111111111, mf.value)

    def test_pca_kurtosis_first_pc(self):
        mf = self.mf["PCAKurtosisFirstPC"](self.X_transformed, self.y)
        self.assertAlmostEqual(-1.72604693348, mf.value)

    def test_pca_skewness_first_pc(self):
        mf = self.mf["PCASkewnessFirstPC"](self.X_transformed, self.y)
        self.assertAlmostEqual(-0.0158262, mf.value)

    def test_calculate_all_metafeatures(self):
        mf = meta_features.calculate_all_metafeatures(
            self.X, self.y, self.categorical, "2")
        self.assertEqual(52, len(mf.metafeature_values))
        sio = StringIO()
        mf.dump(sio)