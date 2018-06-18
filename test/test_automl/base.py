# -*- encoding: utf-8 -*-
import os
import shutil
import time
import unittest
from autosklearn.util.backend import create


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

    def _setUp(self, dir):
        if os.path.exists(dir):
            for i in range(10):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)

    def _create_backend(self, test_name):
        tmp = os.path.join(self.test_dir, '..', '.tmp._%s' % test_name)
        output = os.path.join(self.test_dir, '..', '.output._%s' % test_name)
        # Make sure the folders we wanna create do not already exist.
        self._setUp(tmp)
        self._setUp(output)
        backend = create(tmp, output)
        return backend

    def _tearDown(self, dir):
        """
        Delete the temporary and the output directories manually
        in case they are not deleted.
        """
        if os.path.exists(dir):
            for i in range(10):
                try:
                    shutil.rmtree(dir)
                    break
                except OSError:
                    time.sleep(1)