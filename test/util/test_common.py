# -*- encoding: utf-8 -*-
from functools import partial

import os
import unittest

from autosklearn.util import set_auto_seed, get_auto_seed, del_auto_seed


class TestUtilsCommon(unittest.TestCase):

    def setUp(self):
        self.env_key = 'AUTOSKLEARN_SEED'

    def test_auto_seed(self):
        value = 123
        set_auto_seed(value)
        self.assertEqual(os.environ[self.env_key], str(value))

        del_auto_seed()
        self.assertEqual(os.environ.get(self.env_key), None)

    def test_get_auto_seed(self):
        del_auto_seed()
        self.assertRaises(AssertionError, get_auto_seed)
        set_auto_seed([])
        self.assertRaises(ValueError, get_auto_seed)
        self.assertRaises(ValueError, partial(set_auto_seed, 5))
        del_auto_seed()
        set_auto_seed(5)
        self.assertEqual(os.environ.get(self.env_key), str(5))

if __name__ == '__main__':
    unittest.main()
