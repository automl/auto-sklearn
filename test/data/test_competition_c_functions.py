# -*- encoding: utf-8 -*-
from __future__ import print_function

import os
import unittest

import autosklearn.data_managers.competition_c_functions as competition_c_functions


class CHelperFunctionTest(unittest.TestCase):

    def test_read_sparse_binary_file(self):
        filename = os.path.join(os.path.dirname(__file__),
                                '../.data/dorothea/dorothea_train.data')
        data = competition_c_functions.read_sparse_binary_file(filename, 800,
                                                               100000)

        print(data.indices)
        print(data.indptr)
        print(data.shape)
        print(data)
