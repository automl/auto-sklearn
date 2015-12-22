import unittest

import numpy as np

from autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression import SelectPercentileRegression
from autosklearn.pipeline.util import _test_preprocessing, get_dataset


class SelectPercentileRegressionTest(unittest.TestCase):
    def test_default_configuration(self):
        transformation, original = _test_preprocessing(dataset="boston", Preprocessor=SelectPercentileRegression)
        self.assertEqual(transformation.shape[0], original.shape[0])
        self.assertEqual(transformation.shape[1], int(original.shape[1]/2))
        self.assertFalse((transformation == 0).all())

    def test_preprocessing_dtype(self):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        self.assertEqual(X_train.dtype, np.float32)

        configuration_space = SelectPercentileRegression.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = SelectPercentileRegression(random_state=1,
                                                  **{hp_name: default[hp_name]
                                                     for hp_name in default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset("iris")
        X_train = X_train.astype(np.float64)
        configuration_space = SelectPercentileRegression.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = SelectPercentileRegression(random_state=1,
                                                  **{hp_name: default[hp_name]
                                                     for hp_name in default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)
