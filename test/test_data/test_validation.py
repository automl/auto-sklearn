import unittest
import unittest.mock

from autosklearn.data.validation import InputValidator

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.multiclass import type_of_target


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

    def test_binary_conversion(self):
        validator = InputValidator()

        # Just 2 classes, 1 and 2
        y_train = validator.validate_target(
            np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float64),
            is_classification=True,
        )
        self.assertEqual('binary', type_of_target(y_train))

        # Also make sure that a re-use of the generator is also binary
        y_valid = validator.validate_target(
            np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
            is_classification=True,
        )
        self.assertEqual('binary', type_of_target(y_valid))

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

        with self.assertRaisesRegex(ValueError, "Expected 2D array, got"):
            validator.validate_features({'input1': 1, 'input2': 2})

        with self.assertRaisesRegex(ValueError, "Expected 2D array, got"):
            validator.validate_features(InputValidator())

        X = pd.DataFrame(data=['a', 'b', 'c'], dtype='category')
        with unittest.mock.patch('autosklearn.data.validation.InputValidator._check_and_get_columns_to_encode') as mock_foo:  # noqa E501
            # Mock that all columns are ok. There should be a
            # checker to catch for bugs
            mock_foo.return_value = ([], [])
            with self.assertRaisesRegex(ValueError, 'Failed to convert the input'):
                validator.validate_features(X)

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
