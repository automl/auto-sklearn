import os
import sys
import unittest

from autosklearn.pipeline.components.base import find_components, \
    AutoSklearnClassificationAlgorithm

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_dir)


class TestBase(unittest.TestCase):

    def test_find_components(self):
        c = find_components('dummy_components',
                            os.path.join(this_dir, 'dummy_components'),
                            AutoSklearnClassificationAlgorithm)
        print('COMPONENTS: %s' % repr(c))
        self.assertEqual(len(c), 2)
        self.assertEqual(c['dummy_component_1'].__name__, 'DummyComponent1')
        self.assertEqual(c['dummy_component_2'].__name__, 'DummyComponent2')
