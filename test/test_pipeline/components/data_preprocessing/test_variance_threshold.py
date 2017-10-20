from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.variance_threshold.variance_threshold \
    import VarianceThreshold
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase


class VarianceThresholdTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformations = []
        for i in range(2):
            transformation, original = _test_preprocessing(VarianceThreshold)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1] == transformations[-2]).all())

    def test_default_configuration_sparse_data(self):
        transformations = []
        transformation, original = _test_preprocessing(VarianceThreshold,
                                                       make_sparse=True)
        self.assertEqual(transformation.shape, (100, 3))
        self.assertTrue((transformation.toarray() == original.toarray()[:, 1:]).all())
        self.assertIsInstance(transformation, sparse.csr_matrix)
        transformations.append(transformation)

    def test_preprocessing_dtype(self):
        super(VarianceThresholdTest, self)._test_preprocessing_dtype(
            VarianceThreshold, add_NaNs=False
        )