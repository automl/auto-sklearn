# -*- encoding: utf-8 -*-
import os
import shutil
import time
import unittest


class Base(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(__file__)

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