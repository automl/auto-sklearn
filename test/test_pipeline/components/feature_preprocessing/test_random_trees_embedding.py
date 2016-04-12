import unittest

import numpy as np
import scipy.sparse

from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import \
    RandomTreesEmbedding
from autosklearn.pipeline.util import _test_preprocessing, get_dataset


class RandomTreesEmbeddingComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(RandomTreesEmbedding)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 215)
        self.assertIsInstance(original, np.ndarray)
        self.assertTrue(scipy.sparse.issparse(transformation))
        self.assertTrue(all(transformation.data == 1))

    @unittest.skip("Right now, the RTE returns a float64 array!")
    def test_preprocessing_dtype(self):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        self.assertEqual(X_train.dtype, np.float32)

        configuration_space = RandomTreesEmbedding.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = RandomTreesEmbedding(random_state=1,
                                            **{hp_name: default[hp_name] for
                                               hp_name in
                                               default})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)

        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        X_train = X_train.astype(np.float64)
        configuration_space = RandomTreesEmbedding.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = RandomTreesEmbedding(random_state=1,
                                            **{hp_name: default[hp_name] for
                                               hp_name in
                                               default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)