import unittest

from autosklearn.pipeline.components.base import find_components, \
    AutoSklearnClassificationAlgorithm


class TestBase(unittest.TestCase):

    def test_find_components(self):
        c = find_components('dummy_components', 'dummy_components',
                            AutoSklearnClassificationAlgorithm)
        self.assertEqual(len(c), 2)
        self.assertEqual(c['dummy_component_1'].__name__, 'DummyComponent1')
        self.assertEqual(c['dummy_component_2'].__name__, 'DummyComponent2')
