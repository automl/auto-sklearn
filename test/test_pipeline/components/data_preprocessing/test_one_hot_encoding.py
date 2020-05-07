import unittest

import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.categorical_encoding.\
    one_hot_encoding import OneHotEncoder
from autosklearn.pipeline.components.data_preprocessing.categorical_encoding.\
    no_encoding import NoEncoding
from autosklearn.pipeline.util import _test_preprocessing


def create_X(instances=1000, n_feats=10, categs_per_feat=5, seed=0):
    rs = np.random.RandomState(seed)
    size = (instances, n_feats)
    X = rs.randint(0, categs_per_feat, size=size)
    return X


class OneHotEncoderTest(unittest.TestCase):

    def setUp(self):
        self.X_train = create_X()

    def test_data_type_consistency(self):
        X = np.random.randint(3, 6, (3, 4))
        Y = OneHotEncoder().fit_transform(X)
        self.assertFalse(sparse.issparse(Y))

        X = sparse.csc_matrix(
            ([3, 6, 4, 5], ([0, 1, 2, 1], [3, 2, 1, 0])), shape=(3, 4))
        Y = OneHotEncoder().fit_transform(X)
        self.assertTrue(sparse.issparse(Y))

    def test_default_configuration(self):
        transformations = []
        for i in range(2):
            configuration_space = OneHotEncoder.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            preprocessor = OneHotEncoder(
                random_state=1,
                **{hp_name: default[hp_name] for hp_name in default if default[hp_name] is not None}
                )

            transformer = preprocessor.fit(self.X_train.copy())
            Xt = transformer.transform(self.X_train.copy())
            transformations.append(Xt)
            if len(transformations) > 1:
                np.testing.assert_array_equal(transformations[-1], transformations[-2])

    def test_default_configuration_no_encoding(self):
        transformations = []
        for i in range(2):
            transformation, original = _test_preprocessing(NoEncoding)
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
                self.assertEqual(
                    (transformations[-1] != transformations[-2]).count_nonzero(), 0)

    def test_default_configuration_sparse_no_encoding(self):
        transformations = []

        for i in range(2):
            transformation, original = _test_preprocessing(NoEncoding,
                                                           make_sparse=True)
            self.assertEqual(transformation.shape, original.shape)
            self.assertTrue((transformation.todense() == original.todense()).all())
            transformations.append(transformation)
            if len(transformations) > 1:
                self.assertEqual(
                    (transformations[-1] != transformations[-2]).count_nonzero(), 0)
