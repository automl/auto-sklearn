import unittest

from autosklearn.data.validation import InputValidator

import numpy as np
import pandas as pd
from scipy import sparse


class InputValidatorTest(unittest.TestCase):

    def setUp(self):
        self.X = [
            [2.5, 3.3, 2, 5, 1, 1],
            [1.0, 0.7, 1, 5, 1, 0],
            [1.3, 0.8, 1, 4, 1, 1]
        ]
        self.y = [0, 1, 0]

    def test_list_input(self):
        validator = InputValidator()
        X, y = validator.validate(self.X, self.y)

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_numpy_input(self):
        validator = InputValidator()
        X, y = validator.validate(
            np.array(self.X),
            np.array(self.y)
        )

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsNone(validator.target_encoder)
        self.assertIsNone(validator.feature_encoder)

    def test_sparse_input(self):
        validator = InputValidator()

        # Sparse data
        row_ind = np.array([0, 1, 2])
        col_ind = np.array([1, 2, 1])
        X_sparse = sparse.csr_matrix((np.ones(3), (row_ind, col_ind)))
        X, y = validator.validate(
            X_sparse,
            np.array(self.y)
        )

        self.assertIsInstance(X, sparse.csr.csr_matrix)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsNone(validator.target_encoder)
        self.assertIsNone(validator.feature_encoder)

    def test_dataframe_input_numerical(self):
        for test_type in ['int64', 'float64', 'int8']:
            validator = InputValidator()
            X, y = validator.validate(
                pd.DataFrame(data=self.X, dtype=test_type),
                pd.DataFrame(data=self.y, dtype=test_type),
            )

            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsNone(validator.target_encoder)
            self.assertIsNone(validator.feature_encoder)

    def test_dataframe_input_categorical(self):
        for test_type in ['bool', 'category']:
            validator = InputValidator()
            X, y = validator.validate(
                pd.DataFrame(data=self.X, dtype=test_type),
                pd.DataFrame(data=self.y, dtype=test_type),
                is_classification=True,
            )

            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsNotNone(validator.target_encoder)
            self.assertIsNotNone(validator.feature_encoder)

    def test_dataframe_input_unsupported(self):
        validator = InputValidator()
        with self.assertRaisesRegex(ValueError, "Auto-sklearn does not support time"):
            validator.validate_features(
                pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
            )
        with self.assertRaisesRegex(ValueError, "has invalid type object"):
            validator.validate_features(
                pd.DataFrame({'string': ['foo']})
            )

    def test_dataframe_econding_1D(self):
        validator = InputValidator()
        y = validator.validate_target(
            pd.DataFrame(data=self.y, dtype=bool),
            is_classification=True,
        )
        np.testing.assert_array_almost_equal(np.array([0, 1, 0]), y)

        # Result should not change on a multi call
        y = validator.validate_target(y)
        np.testing.assert_array_almost_equal(np.array([0, 1, 0]), y)

        y_decoded = validator.decode_target(y)
        np.testing.assert_array_almost_equal(np.array(self.y, dtype=bool), y_decoded)

        # Now go with categorical data
        validator = InputValidator()
        y = validator.validate_target(
            pd.DataFrame(data=['a', 'a', 'b', 'c', 'a'], dtype='category'),
            is_classification=True,
        )
        np.testing.assert_array_almost_equal(np.array([0, 0, 1, 2, 0]), y)

        y_decoded = validator.decode_target(y)
        self.assertListEqual(['a', 'a', 'b', 'c', 'a'], y_decoded.tolist())

    def test_dataframe_econding_2D(self):
        validator = InputValidator()
        multi_label = pd.DataFrame(
            np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
            dtype=bool
        )
        y = validator.validate_target(multi_label, is_classification=True)

        # Result should not change on a multi call
        y_new = validator.validate_target(multi_label)
        np.testing.assert_array_almost_equal(y_new, y)

        y_decoded = validator.decode_target(y)
        np.testing.assert_array_almost_equal(y, y_decoded)
