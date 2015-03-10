import unittest

import numpy
import scipy.sparse

from ParamSklearn.components.preprocessing.random_trees_embedding import \
    RandomTreesEmbedding
from ParamSklearn.util import _test_preprocessing


class RandomTreesEmbeddingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(RandomTreesEmbedding)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 213)
        self.assertIsInstance(original, numpy.ndarray)
        self.assertTrue(scipy.sparse.issparse(transformation))
        self.assertTrue(all(transformation.data == 1))