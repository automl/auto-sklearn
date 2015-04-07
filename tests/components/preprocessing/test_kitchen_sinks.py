import unittest

import numpy as np

from ParamSklearn.components.preprocessing.kitchen_sinks import RandomKitchenSinks
from ParamSklearn.util import _test_preprocessing, get_dataset


class KitchenSinkComponent(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(RandomKitchenSinks)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 100)
        self.assertFalse((transformation == 0).all())

    @unittest.skip("Right now, the RBFSampler returns a float64 array!")
    def _test_preprocessing_dtype(self):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        self.assertEqual(X_train.dtype, np.float32)

        configuration_space = RandomKitchenSinks.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = RandomKitchenSinks(random_state=1,
                                    **{hp.hyperparameter.name: hp.value for hp
                                       in
                                       default.values.values()})
        preprocessor.fit(X_train)
        print id(X_train)
        Xt = preprocessor.transform(X_train)
        print id(Xt)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        X_train = X_train.astype(np.float64)
        configuration_space = RandomKitchenSinks.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = RandomKitchenSinks(random_state=1,
                                    **{hp.hyperparameter.name: hp.value for hp
                                       in
                                       default.values.values()})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)

        # Sparse
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris", make_sparse=True)
        self.assertEqual(X_train.dtype, np.float32)
        configuration_space = RandomKitchenSinks.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = RandomKitchenSinks(random_state=1,
                                    **{hp.hyperparameter.name: hp.value for hp
                                       in
                                       default.values.values()})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris", make_sparse=True)
        X_train = X_train.astype(np.float64)
        configuration_space = RandomKitchenSinks.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = RandomKitchenSinks(random_state=1,
                                    **{hp.hyperparameter.name: hp.value for hp
                                       in
                                       default.values.values()})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)
