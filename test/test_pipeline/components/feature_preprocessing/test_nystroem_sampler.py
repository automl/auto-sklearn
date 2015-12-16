import unittest

import numpy as np
import sklearn.preprocessing

from autosklearn.pipeline.components.feature_preprocessing.nystroem_sampler import \
    Nystroem
from autosklearn.pipeline.util import _test_preprocessing, get_dataset


class NystroemComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(Nystroem)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 100)
        self.assertFalse((transformation == 0).all())

        # Custon preprocessing test to check if clipping to zero works
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        original_X_train = X_train.copy()
        ss = sklearn.preprocessing.StandardScaler()
        X_train = ss.fit_transform(X_train)
        configuration_space = Nystroem.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()

        preprocessor = Nystroem(random_state=1,
                                **{hp_name: default[hp_name]
                                   for hp_name in default
                                   if default[hp_name] is not None})

        transformer = preprocessor.fit(X_train, Y_train)
        transformation, original = transformer.transform(
            X_train), original_X_train
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 100)

    #@unittest.skip("Right now, the RBFSampler returns a float64 array!")
    def _test_preprocessing_dtype(self):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        self.assertEqual(X_train.dtype, np.float32)

        configuration_space = Nystroem.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = Nystroem(random_state=1,
                                          **{hp.hyperparameter.name: hp.value
                                             for hp
                                             in
                                             default.values.values()})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        X_train = X_train.astype(np.float64)
        configuration_space = Nystroem.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = Nystroem(random_state=1,
                                          **{hp.hyperparameter.name: hp.value
                                             for hp
                                             in
                                             default.values.values()})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)

        # Sparse
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris", make_sparse=True)
        self.assertEqual(X_train.dtype, np.float32)
        configuration_space = Nystroem.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = Nystroem(random_state=1,
                                          **{hp.hyperparameter.name: hp.value
                                             for hp
                                             in
                                             default.values.values()})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris", make_sparse=True)
        X_train = X_train.astype(np.float64)
        configuration_space = Nystroem.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = Nystroem(random_state=1,
                                          **{hp.hyperparameter.name: hp.value
                                             for hp
                                             in
                                             default.values.values()})
        preprocessor.fit(X_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)
