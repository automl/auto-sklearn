import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.feature_type_numerical import (
    NumericalPreprocessingPipeline,
)

import unittest


class NumericalPreprocessingPipelineTest(unittest.TestCase):
    def test_data_type_consistency(self):
        X = np.random.rand(3, 4)
        Y = NumericalPreprocessingPipeline(
            feat_type={0: "numerical", 1: "numerical", 2: "numerical"}
        ).fit_transform(X)
        self.assertFalse(sparse.issparse(Y))

        X = sparse.csc_matrix(
            ([3.0, 6.0, 4.0, 5.0], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4)
        )
        Y = NumericalPreprocessingPipeline(
            feat_type={0: "numerical", 1: "numerical", 2: "numerical"}
        ).fit_transform(X)
        self.assertTrue(sparse.issparse(Y))

    def test_fit_transform(self):
        X = np.array(
            [[3.14, 1.0, 1.0], [3.14, 2.0, np.nan], [3.14, 3.0, 3.0]]
        )  # noqa : matrix legibility
        # 1st column should be droped due to low variance
        # The 2nd should be be standardized (default rescaling algorithm)
        # The 3rd will get a value imputed by the mean (2.), therefore the
        # transformation here will have the same effect as on the the 2nd column
        sdev = np.sqrt(2 / 3)
        Y1 = np.array(
            [
                [-1 / sdev, -1 / sdev],
                [0.0, 0.0],  # noqa : matrix legibility
                [1 / sdev, 1 / sdev],
            ]
        )  # noqa : matrix legibility
        # dense input
        Yt = NumericalPreprocessingPipeline(
            feat_type={0: "numerical", 1: "numerical", 2: "numerical"}
        ).fit_transform(X)
        np.testing.assert_array_almost_equal(Yt, Y1)
        # sparse input (uses with_mean=False)
        Y2 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) / sdev
        X_sparse = sparse.csc_matrix(X)
        Yt = NumericalPreprocessingPipeline(
            feat_type={0: "numerical", 1: "numerical", 2: "numerical"}
        ).fit_transform(X_sparse)
        np.testing.assert_array_almost_equal(Yt.todense(), Y2)

    def test_transform(self):
        X1 = np.array(
            [[3.14, 1.0, 1.0], [3.14, 2.0, np.nan], [3.14, 3.0, 3.0]]
        )  # noqa : matrix legibility
        sdev = np.sqrt(2 / 3)
        # fit
        NPP = NumericalPreprocessingPipeline(
            feat_type={0: "numerical", 1: "numerical", 2: "numerical"}
        )
        NPP.fit_transform(X1)
        # transform
        X2 = np.array([[1.0, 5.0, 8.0], [2.0, 6.0, 9.0], [3.0, 7.0, np.nan]])
        Yt = NPP.transform(X2)
        # imputation, variance_threshold and rescaling are done using the data already
        # fitted, therefore:
        Y2 = np.array(
            [[3 / sdev, 6 / sdev], [4 / sdev, 7 / sdev], [5 / sdev, 0.0]]
        )  # noqa : matrix legibility
        np.testing.assert_array_almost_equal(Yt, Y2)
