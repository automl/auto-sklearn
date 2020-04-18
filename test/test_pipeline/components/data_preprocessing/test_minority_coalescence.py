import unittest
import numpy as np

import scipy.sparse

from autosklearn.pipeline.components.data_preprocessing.minority_coalescense\
    .minority_coalescer import MinorityCoalescer
from autosklearn.pipeline.components.data_preprocessing.minority_coalescense\
    .no_coalescense import NoCoalescence


class MinorityCoalescerTest(unittest.TestCase):

    def test_data_type_consistency(self):
        X = np.random.randint(3, 6, (3, 4))
        Y = MinorityCoalescer().fit_transform(X)
        self.assertFalse(scipy.sparse.issparse(Y))

        X = scipy.sparse.csc_matrix(
            ([3, 6, 4, 5], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4))
        Y = MinorityCoalescer().fit_transform(X)
        self.assertTrue(scipy.sparse.issparse(Y))

    def test_no_coalescence(self):
        X = np.random.randint(0, 255, (3, 4))
        Y = NoCoalescence().fit_transform(X)
        np.testing.assert_array_almost_equal(Y, X)
        # Assert no copies were made
        self.assertEqual(id(X), id(Y))
