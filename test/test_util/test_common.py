# -*- encoding: utf-8 -*-
import os

from autosklearn.util.common import check_pid

import unittest


class TestUtilsCommon(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_check_pid(self):
        our_pid = os.getpid()

        exists = check_pid(our_pid)
        self.assertTrue(exists)
        our_pid = -11000  # We hope this pid does not exist
        exists = check_pid(our_pid)
        self.assertFalse(exists)

    def test_check_pid_invalid(self):
        # Test with a PID that is guaranteed not to exist
        # Using a very large number that is unlikely to be a valid PID
        invalid_pid = 999999999
        exists = check_pid(invalid_pid)
        self.assertFalse(exists)

    def test_check_pid_zero(self):
        # Test with PID 0, which is typically invalid for user processes
        exists = check_pid(0)
        self.assertFalse(exists)


if __name__ == "__main__":
    unittest.main()
