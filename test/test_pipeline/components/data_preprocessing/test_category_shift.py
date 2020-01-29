import unittest
import numpy as np
import scipy.sparse

from autosklearn.pipeline.components.data_preprocessing.category_shift.\
    category_shift import CategoryShift


class CategoryShiftTest(unittest.TestCase):

    def test_data_type_consistency(self):
        X = np.random.randint(0, 255, (3, 4))
        Y = CategoryShift().fit_transform(X)
        self.assertFalse(scipy.sparse.issparse(Y))

        X = scipy.sparse.csc_matrix(
            ([1, 2, 0, 4], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4))
        Y = CategoryShift().fit_transform(X)
        self.assertTrue(scipy.sparse.issparse(Y))
