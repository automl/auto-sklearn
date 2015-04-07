import unittest

import numpy as np

from ParamSklearn.components.preprocessing.pca import PCA
from ParamSklearn.util import _test_preprocessing, get_dataset


class PCAComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformations = []
        for i in range(10):
            transformation, original = _test_preprocessing(PCA)
            self.assertEqual(transformation.shape, original.shape)
            self.assertFalse((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue((transformations[-1] == transformations[-2]).all())

    def test_preprocessing_dtype(self):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        self.assertEqual(X_train.dtype, np.float32)

        configuration_space = PCA.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = PCA(random_state=1,
                                    **{hp.hyperparameter.name: hp.value for hp
                                       in
                                       default.values.values()})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        X_train = X_train.astype(np.float64)
        configuration_space = PCA.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = PCA(random_state=1,
                                    **{hp.hyperparameter.name: hp.value for hp
                                       in
                                       default.values.values()})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)