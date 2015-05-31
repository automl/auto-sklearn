import os
import unittest

import autosklearn.data.chelper_functions as chelper_function


class CHelperFunctionTest(unittest.TestCase):
    def test_read_sparse_binary_file(self):
        filename = os.path.join(os.path.dirname(__file__),
                                "../.data/dorothea/dorothea_train.data")
        data = chelper_function.read_sparse_binary_file(
            filename, 800, 100000)

        print data.indices
        print data.indptr
        print data.shape
        print data