import os
import unittest

import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.one_hot_encoding.one_hot_encoding \
    import OneHotEncoder
from autosklearn.pipeline.util import _test_preprocessing


def create_X(instances=1000, n_feats=10, categs_per_feat=5, random_state=0):
    np.random.seed(random_state)
    size = (instances, n_feats)
    X = np.random.randint(0, categs_per_feat, size=size)
    return X

class OneHotEncoderTest(unittest.TestCase):

    def setUp(self):
        self.X_train = create_X()

    def test_default_configuration(self):
        transformations = []
        for i in range(2):
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            preprocessor = OneHotEncoder(random_state=1,
                                         **{hp_name: default[hp_name] for hp_name in
                                           default if default[hp_name] is not None})

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                self.assertFalse(
                    (transformations[-1] != transformations[-2]).all())

    def test_default_configuration_no_encoding(self):
        transformations = []
        for i in range(2):
            transformation, original = _test_preprocessing(OneHotEncoder)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation == original).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1] == transformations[-2]).all())

    def test_default_configuration_sparse_data(self):
        transformations = []

        self.X_train[~np.isfinite(self.X_train)] = 0
        self.X_train = sparse.csc_matrix(self.X_train)

        for i in range(2):
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            preprocessor = OneHotEncoder(random_state=1,
                                         **{hp_name: default[hp_name] for
                                            hp_name in
                                            default if
                                            default[hp_name] is not None})

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                self.assertFalse(
                    (transformations[-1].todense() != transformations[
                        -2].todense()).all())

    def test_default_configuration_sparse_no_encoding(self):
        transformations = []

        for i in range(2):
            transformation, original = _test_preprocessing(OneHotEncoder,
                                                           make_sparse=True)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation.todense() == original.todense()).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertTrue(
                    (transformations[-1].todense() == transformations[-2].todense()).all())
