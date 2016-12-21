import unittest

import numpy as np
from sklearn.datasets import make_classification

from autosklearn.pipeline.components.feature_preprocessing.gaussian_random_projection import \
    GaussRandomProjection


class GaussianRandomProjectionComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        X_train, Y_train = make_classification(n_samples=1000, n_classes=2,
                                               n_features=1000)

        original_X_train = X_train.copy()
        configuration_space = GaussRandomProjection.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()

        preprocessor = GaussRandomProjection(
            random_state=np.random.RandomState(1),
            **{hp_name: default[hp_name] for hp_name in
               default if default[hp_name] is not None})

        transformer = preprocessor.fit(X_train, Y_train)
        transformation, original = transformer.transform(X_train), original_X_train

        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 331)
        self.assertFalse((transformation == 0).all())
