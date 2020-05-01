from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.imputation.numerical_imputation\
    import NumericalImputation
from autosklearn.pipeline.util import _test_preprocessing, PreprocessingTestCase


class NumericalImputationTest(PreprocessingTestCase):
    def test_default_configuration(self):
        transformations = []
        for i in range(2):
            transformation, original = _test_preprocessing(NumericalImputation)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1] == transformations[-2]).all())

    def test_default_configuration_sparse_data(self):
        transformations = []
        transformation, original = _test_preprocessing(NumericalImputation,
                                                       make_sparse=True)
        self.assertEqual(transformation.shape, original.shape)
        self.assertTrue((transformation.data == original.data).all())
        self.assertIsInstance(transformation, sparse.csc_matrix)
        transformations.append(transformation)

    def test_preprocessing_dtype(self):
        super(NumericalImputationTest, self)._test_preprocessing_dtype(
            NumericalImputation, add_NaNs=True)
