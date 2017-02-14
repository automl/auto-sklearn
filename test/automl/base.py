# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import shutil
import time
import unittest


class Base(unittest.TestCase):
    _multiprocess_can_split_ = True
    """All tests which are a subclass of this must define their own output
    directory and call self._setUp."""

    def setUp(self):
        self.test_dir = os.path.dirname(__file__)

        try:
            travis = os.environ['TRAVIS']
            self.travis = True
        except Exception:
            self.travis = False

    def _setUp(self, output):
        if os.path.exists(output):
            for i in range(10):
                try:
                    shutil.rmtree(output)
                    break
                except OSError:
                    time.sleep(1)
        try:
            os.makedirs(output)
        except OSError:
            pass


    def _tearDown(self, output):
        if os.path.exists(output):
            for i in range(10):
                try:
                    shutil.rmtree(output)
                    break
                except OSError:
                    time.sleep(1)