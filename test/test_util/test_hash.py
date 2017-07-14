import unittest

import numpy as np
import scipy.sparse

from autosklearn.util.hash import hash_array_or_matrix


class HashTests(unittest.TestCase):

    def test_c_contiguous_array(self):
        array = np.array([[1, 2], [3, 4]])

        hash = hash_array_or_matrix(array)

        self.assertIsNotNone(hash)

    def test_f_contiguous_array(self):
        array = np.array([[1, 2], [3, 4]])
        array = np.asfortranarray(array)

        hash = hash_array_or_matrix(array)

        self.assertIsNotNone(hash)

    def test_transpose_arrays(self):
        c_array = np.array([[1, 2], [3, 4]])
        f_array = np.array([[1, 3], [2, 4]])
        f_array = np.asfortranarray(f_array)

        c_hash = hash_array_or_matrix(c_array)
        f_hash = hash_array_or_matrix(f_array)

        self.assertEqual(c_hash, f_hash)

    def test_same_data_arrays(self):
        first_array = np.array([[1, 2], [3, 4]])
        second_array = np.array([[1, 2], [3, 4]])

        first_hash = hash_array_or_matrix(first_array)
        second_hash = hash_array_or_matrix(second_array)

        self.assertEqual(first_hash, second_hash)

    def test_different_data_arrays(self):
        first_array = np.array([[1, 2], [3, 4]])
        second_array = np.array([[1, 3], [2, 4]])

        first_hash = hash_array_or_matrix(first_array)
        second_hash = hash_array_or_matrix(second_array)

        self.assertNotEqual(first_hash, second_hash)

    def test_scipy_csr(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        hash = hash_array_or_matrix(matrix)

        self.assertIsNotNone(hash)