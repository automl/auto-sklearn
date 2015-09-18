# -*- encoding: utf-8 -*-
"""Created on Dec 16, 2014.

@author: Katharina Eggensperger
@projekt: AutoML2015

"""

import unittest
import sys

from autosklearn.util.io import *


class TestIO(unittest.TestCase):

    def test_check_existing(self):
        our_pid = os.getpid()
        exists = check_pid(our_pid)
        self.assertTrue(exists)
        our_pid = -11000  # We hope this pid does not exist
        exists = check_pid(our_pid)
        self.assertFalse(exists)

    def test_search_prog(self):
        if sys.platform == 'linux2':
            self.assertEqual(search_prog('bash'), '/bin/bash')
            self.assertEqual(search_prog('dfgsdfgmsadgksadg'), None)

    def test_find_files(self):
        if sys.platform == 'linux2':
            self.assertEqual(find_files('/bin', 'bash'), ['/bin/bash'])
            self.assertNotEqual(find_files('/bin', 'basfdsfsadfsadfdash'),
                                ['/bin/bash'])
            self.assertEqual(find_files('/bin', 'basfdsfsadfsadfdash'), [])



if __name__ == '__main__':
    unittest.main()
