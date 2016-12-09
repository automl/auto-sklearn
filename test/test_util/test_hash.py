import unittest

import numpy as np

from autosklearn.util.hash import hash_numpy_array


class HashTests(unittest.TestCase):

    def test_c_contiguous_array(self):
        array = np.array([[1, 2], [3, 4]])

        hash = hash_numpy_array(array)

        self.assertIsNotNone(hash)

    def test_f_contiguous_array(self):
        array = np.array([[1, 2], [3, 4]])
        array = np.asfortranarray(array)

        hash = hash_numpy_array(array)

        self.assertIsNotNone(hash)

    def test_transpose_arrays(self):
        c_array = np.array([[1, 2], [3, 4]])
        f_array = np.array([[1, 3], [2, 4]])
        f_array = np.asfortranarray(f_array)

        c_hash = hash_numpy_array(c_array)
        f_hash = hash_numpy_array(f_array)

        self.assertEqual(c_hash, f_hash)

    def test_same_data_arrays(self):
        first_array = np.array([[1, 2], [3, 4]])
        second_array = np.array([[1, 2], [3, 4]])

        first_hash = hash_numpy_array(first_array)
        second_hash = hash_numpy_array(second_array)

        self.assertEqual(first_hash, second_hash)

    def test_different_data_arrays(self):
        first_array = np.array([[1, 2], [3, 4]])
        second_array = np.array([[1, 3], [2, 4]])

        first_hash = hash_numpy_array(first_array)
        second_hash = hash_numpy_array(second_array)

        self.assertNotEqual(first_hash, second_hash)