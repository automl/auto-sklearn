import unittest

from autosklearn.data.validation import InputValidator

import numpy as np
import pandas as pd


class InputValidatorTest(unittest.TestCase):

    def setUp(self):
        self.X = [
            [2.5, 3.3, 2, 5, 1, 1],
            [1.0, 0.7, 1, 5, 1, 0],
            [1.3, 0.8, 1, 4, 1, 1]
        ]
        self.y = [0, 1, 0]

    def test_array_input(self):
        validator = InputValidator()
        X, y = validator.validate(self.X, self.y)

        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

    def test_numpy_input(self):
        validator = InputValidator()
        X, y = validator.validate(
            np.array(self.X),
            np.array(self.y)
        )

        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertTrue(validator.target_encoder is None)
        self.assertTrue(validator.feature_encoder is None)

    def test_dataframe_input_numerical(self):
        for test_type in ['int64', 'float64', 'int8']:
            validator = InputValidator()
            X, y = validator.validate(
                pd.DataFrame(data=self.X, dtype=test_type),
                pd.DataFrame(data=self.y, dtype=test_type),
            )

            self.assertTrue(isinstance(X, np.ndarray))
            self.assertTrue(isinstance(y, np.ndarray))
            self.assertTrue(validator.target_encoder is None)
            self.assertTrue(validator.feature_encoder is None)

    def test_dataframe_input_categorical(self):
        for test_type in ['bool', 'category']:
            validator = InputValidator()
            X, y = validator.validate(
                pd.DataFrame(data=self.X, dtype=test_type),
                pd.DataFrame(data=self.y, dtype=test_type),
            )

            self.assertTrue(isinstance(X, np.ndarray))
            self.assertTrue(isinstance(y, np.ndarray))
            self.assertTrue(validator.target_encoder is not None)
            self.assertTrue(validator.feature_encoder is not None)

    def test_dataframe_input_unsupported(self):
        validator = InputValidator()
        with self.assertRaises(ValueError):
            validator.validate_features(
                pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
            )
        with self.assertRaises(ValueError):
            validator.validate_features(
                pd.DataFrame({'string': ['foo']})
            )
        with self.assertRaises(ValueError):
            validator.validate_features(
                pd.DataFrame({'object': InputValidator()})
            )

    def test_dataframe_econding(self):
        validator = InputValidator()
        y = validator.validate_target(
            pd.DataFrame(data=self.y, dtype=bool),
        )
        np.testing.assert_array_almost_equal(np.array([0, 1, 0]), y)

        # Result should not change on a multi call
        y = validator.validate_target(y)
        np.testing.assert_array_almost_equal(np.array([0, 1, 0]), y)

        y_decoded = validator.decode(y)
        np.testing.assert_array_almost_equal(np.array(self.y, dtype=bool), y_decoded)
