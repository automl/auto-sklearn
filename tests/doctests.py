import doctest
import os
import unittest

import AutoSklearn


class DocumentationTest(unittest.TestCase):
    def test_first_steps(self):
        filename = os.path.dirname(AutoSklearn.__file__)
        filename = os.path.join(filename, "..", "source", "first_steps.rst")
        failed, run = doctest.testfile(filename, module_relative=False)
        self.assertEqual(0, failed)