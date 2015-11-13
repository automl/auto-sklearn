# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import unittest

import autosklearn.data.competition_c_functions as competition_c_functions

class CHelperFunctionTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_read_sparse_file(self):
        print("test_read_sparse_file")
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/newsgroups/newsgroups_valid.data')
        data = competition_c_functions.read_sparse_file(
            filename, 1877, 61188, max_memory_in_mb=0.01)
        print(data.shape)
        print(data.nnz)

    def test_read_sparse_binary_file(self):
        print("test_read_sparse_binary_file")
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/dorothea/dorothea_train.data')
        data = competition_c_functions.read_sparse_binary_file(filename, 800,
                                                               100000)
        #print(data)

    def test_read_dense(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/31_bac/31_bac_train.data')
        data = competition_c_functions.read_dense_file(
            filename, 670, 20, 0.01)
        self.assertEqual(data.shape, (131, 20))

unittest.main()
