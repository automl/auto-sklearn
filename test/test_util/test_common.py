# -*- encoding: utf-8 -*-
import os
import unittest

from autosklearn.util import check_pid


class TestUtilsCommon(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_check_pid(self):
        our_pid = os.getpid()

        exists = check_pid(our_pid)
        self.assertTrue(exists)
        our_pid = -11000  # We hope this pid does not exist
        exists = check_pid(our_pid)
        self.assertFalse(exists)

if __name__ == '__main__':
    unittest.main()
