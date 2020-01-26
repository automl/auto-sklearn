import unittest
import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor


class PreprocessingPipelineTest(unittest.TestCase):

    def do_a_fit_transform(self, sparse_input):
        # X will be the input and Y is what we expect after transform. categ_feat stores
        # indicators of feature type (True if categorical, False if numerical)
        X, Y, categ_feat = [], [], []
        # Feature 1 (numerical):
        # This feature should be dropped due to lack of variance.
        categ_feat.append(False)
        X.append(np.array([3.14, 3.14, 3.14]).reshape(3, 1))
        Y.append(np.array([]).reshape(3, 0))
        # Feature 2 (numerical):
        # This feature should be normalized by having its mean subtracted from all
        # elements and by having them divided by the standard deviation.
        categ_feat.append(False)
        nf = np.array([1., 2., 3.]).reshape(3, 1)  # mean = 2.
        sdev = np.sqrt(2. / 3.)
        shift = 0 if sparse_input else 2.  # if sparse_input, there is no mean subtraction
        nft = (nf - shift) / sdev
        X.append(nf)
        Y.append(nft)
        # Feature 3 (numerical):
        # This feature has a missing value that should be imputed by the mean of the other
        # values (2.). This feature should also be normalized as in the previous feature.
        categ_feat.append(False)
        X.append(np.array([1., np.nan, 3.]).reshape(3, 1))
        Y.append(nft.copy())
        # Feature 4 (categorical)
        # This feature should be one hot encoded.
        categ_feat.append(True)
        X.append(np.array([1, 3, 2]).reshape(3, 1))
        Y.append(np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]))
        # Feature 5 (categorical)
        # This feature should be one hot encoded. (A discontinuous category set or
        # a category 0 shouldn't be problems.)
        categ_feat.append(True)
        X.append(np.array([2, 1, 9]).reshape(3, 1))
        Y.append(np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]]))
        # Feature 6 (categorical)
        # This feature should be one hot encoded. The missing value gets imputed as
        # a category on its own.
        categ_feat.append(True)
        X.append(np.array([1, 1, np.nan]).reshape(3, 1))
        Y.append(np.array([
            [0, 1],
            [0, 1],
            [1, 0]]))
        # Combine datasets and shuffle columns:
        n_feats = len(categ_feat)
        random_order = np.random.choice(np.arange(n_feats), size=n_feats, replace=False)
        # Shuffle cat_feat according to random_order
        categ_feat = np.array(categ_feat)[random_order]
        # Shuffle X according to random_order
        X = np.array(X)[random_order]
        X_comb = np.hstack(X)
        # Shuffle Y according to random_order and reorder it as the PreprocessingPipeline
        # does (i.e. categorical features come first in Y).
        num_feat = np.logical_not(categ_feat)
        y_order = random_order[np.argsort(num_feat)]
        Y = [Y[n] for n in y_order]
        Y_comb = np.hstack(Y)
        # Data preprocessing
        DPP = DataPreprocessor(categorical_features=categ_feat)
        X_comb = sparse.csc_matrix(X_comb) if sparse_input else X_comb
        Y_comb_out_1 = DPP.fit_transform(X_comb)
        # Check if Y_comb_out is what we expect it to be:
        self.assertEqual(sparse_input, sparse.issparse(Y_comb_out_1))
        Y_comb_out_1 = Y_comb_out_1.todense() if sparse_input else Y_comb_out_1
        np.testing.assert_array_almost_equal(Y_comb_out_1, Y_comb)
        # Do it again, but using the already fitted pipeline
        Y_comb_out_2 = DPP.transform(X_comb)
        # Consistency check
        self.assertEqual(sparse_input, sparse.issparse(Y_comb_out_2))
        Y_comb_out_2 = Y_comb_out_2.todense() if sparse_input else Y_comb_out_2
        np.testing.assert_array_equal(Y_comb_out_1, Y_comb_out_2)

    def test_fit_transform(self):
        self.do_a_fit_transform(sparse_input=False)

    def test_fit_transform_sparse(self):
        self.do_a_fit_transform(sparse_input=True)

    def test_string_categories(self):
        # Numerical dataset (as used in NumericalPreprocessingPipelineTest)
        X_num = np.array([
            [3.14, 1.,     1.],   # noqa : matrix legibility
            [3.14, 2., np.nan],   # noqa : matrix legibility
            [3.14, 3.,     3.]])  # noqa : matrix legibility
        # Categorical string dataset
        X_cat = np.array([
            ['red', 'medium', 'small'],
            ['blue', 'short', 'big'],
            ['white', 'tall', np.nan]])
        # Combined dataset with shuffled columns:
        X_comb = np.hstack((X_num, X_cat))
        categ_feat = np.array([False] * 3 + [True] * 3)
        random_order = np.random.choice(np.arange(6), size=6, replace=False)
        X_comb = X_comb[:, random_order]
        categ_feat = categ_feat[random_order]
        # Strings are not allowed, therefore:
        with self.assertRaises(ValueError):
            DataPreprocessor(categorical_features=categ_feat).fit_transform(X_comb)
