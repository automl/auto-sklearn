# -*- encoding: utf-8 -*-
import os
import unittest

import autosklearn.data.competition_c_functions as competition_c_functions


class CHelperFunctionTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_read_sparse_file(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/newsgroup/newsgroups_valid.data')
        data = competition_c_functions.read_sparse_file(
            filename, 1877, 61188, max_memory_in_mb=0.01)
        self.assertEqual(data.nnz, 8192)
        data = competition_c_functions.read_sparse_file(
            filename, 1877, 61188)
        self.assertEqual(data.nnz, 246216)

    def test_read_sparse_binary_file(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/dorothea/dorothea_train.data')
        data = competition_c_functions.read_sparse_binary_file(
            filename, 800, 100000)
        self.assertEqual(data.nnz, 727760)
        data = competition_c_functions.read_sparse_binary_file(
            filename, 800, 100000, max_memory_in_mb=0.01)
        self.assertEqual(data.nnz, 8192)

    def test_read_dense(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/31_bac/31_bac_train.data')
        data = competition_c_functions.read_dense_file(
            filename, 670, 20, 0.01)
        self.assertEqual(data.shape, (131, 20))

if __name__ == "__main__":
    unittest.main()
