import unittest

import numpy as np
import scipy.sparse
import sklearn.preprocessing

from autosklearn.pipeline.components.feature_preprocessing.select_rates_regression import \
    SelectRegressionRates
from autosklearn.pipeline.util import _test_preprocessing, get_dataset


class SelectRegressionRatesComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(SelectRegressionRates)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 4)
        self.assertFalse((transformation == 0).all())

        transformation, original = _test_preprocessing(
            SelectRegressionRates, make_sparse=True)
        self.assertTrue(scipy.sparse.issparse(transformation))
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], int(original.shape[1] / 2))

        # Makes sure that the features are reduced, not the number of samples
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        original_X_train = X_train.copy()
        ss = sklearn.preprocessing.StandardScaler()
        X_train = ss.fit_transform(X_train)
        configuration_space = SelectRegressionRates.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()

        preprocessor = SelectRegressionRates(random_state=1,
                                             **{hp_name: default[hp_name]
                                                for hp_name in default
                                                if default[hp_name] is not None})

        transformer = preprocessor.fit(X_train, Y_train)
        transformation, original = transformer.transform(
            X_train), original_X_train
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], 21)

    def test_default_configuration_regression(self):
        transformation, original = _test_preprocessing(
            SelectRegressionRates,
            dataset='boston',
        )
        self.assertEqual(transformation.shape[0], original.shape[0])
        # From 13 to 12 features
        self.assertEqual(transformation.shape[1], 12)
        self.assertFalse((transformation == 0).all())

    def test_preprocessing_dtype_regression(self):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("boston")
        self.assertEqual(X_train.dtype, np.float32)

        dataset_properties = {'target_type': 'regression'}

        configuration_space = SelectRegressionRates.get_hyperparameter_search_space(
            dataset_properties
        )
        default = configuration_space.get_default_configuration()
        preprocessor = SelectRegressionRates(random_state=1,
                                             **{hp_name: default[hp_name] for hp_name in
                                                default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("boston")
        X_train = X_train.astype(np.float64)
        configuration_space = SelectRegressionRates.get_hyperparameter_search_space(
            dataset_properties
        )
        default = configuration_space.get_default_configuration()
        preprocessor = SelectRegressionRates(random_state=1,
                                             **{hp_name: default[hp_name] for hp_name in
                                                default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)
