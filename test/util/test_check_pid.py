# -*- encoding: utf-8 -*-
"""Created on Dec 16, 2014.

@author: Katharina Eggensperger
@projekt: AutoML2015

"""

import os
import unittest

from autosklearn.util import check_pid


class TestCheckPID(unittest.TestCase):

    def test_check_existing(self):
        our_pid = os.getpid()
        exists = check_pid(our_pid)
        self.assertTrue(exists)
        our_pid = -11000  # We hope this pid does not exist
        exists = check_pid(our_pid)
        self.assertFalse(exists)


if __name__ == '__main__':
    unittest.main()
