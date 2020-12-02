import itertools

import unittest
import unittest.mock

import numpy as np

import pandas as pd

from scipy import sparse

import sklearn.datasets
import sklearn.model_selection
from sklearn.utils.multiclass import type_of_target

from autosklearn.data.validation import InputValidator


class InputValidatorTest(unittest.TestCase):

    def setUp(self):
        self.X = [
            [2.5, 3.3, 2, 5, 1, 1],
            [1.0, 0.7, 1, 5, 1, 0],
            [1.3, 0.8, 1, 4, 1, 1]
        ]
        self.y = [0, 1, 0]

    def test_list_input(self):
        """
        Makes sure that a list is converted to nparray
        """
        validator = InputValidator()
        X, y = validator.validate(self.X, self.y)

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)

    def test_numpy_input(self):
        """
        Makes sure that no encoding is needed for a
        numpy float object. Also test features/target
        validation methods
        """
        validator = InputValidator()
        X = validator.validate_features(
            np.array(self.X),
        )
        y = validator.validate_target(
            np.array(self.y)
        )

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsNone(validator.target_encoder)
        self.assertIsNone(validator.feature_encoder)

    def test_sparse_numpy_input(self):
        """
        Makes sure that no encoder is needed when
        working with sparse float data
        """
        validator = InputValidator()

        # Sparse data
        row_ind = np.array([0, 1, 2])
        col_ind = np.array([1, 2, 1])
        X_sparse = sparse.csr_matrix((np.ones(3), (row_ind, col_ind)))
        X = validator.validate_features(
            X_sparse,
        )
        y = validator.validate_target(
            np.array(self.y)
        )

        self.assertIsInstance(X, sparse.csr.csr_matrix)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsNone(validator.target_encoder)
        self.assertIsNone(validator.feature_encoder)

        # Sparse targets should not be supported
        data = np.array([1, 2, 3, 4, 5, 6])
        col = np.array([0, 0, 0, 0, 0, 0])
        row = np.array([0,  2,  3,  6,  7, 10])
        y = sparse.csr_matrix((data, (row, col)), shape=(11, 1))
        with self.assertRaisesRegex(ValueError, 'scipy.sparse.csr_matrix.todense'):
            validator = InputValidator().validate_target(y)

    def test_dataframe_input_numerical(self):
        """
        Makes sure that we don't encode numerical data
        """
        for test_type in ['int64', 'float64', 'int8']:
            validator = InputValidator()
            X = validator.validate_features(
                pd.DataFrame(data=self.X, dtype=test_type),
            )
            y = validator.validate_target(
                pd.DataFrame(data=self.y, dtype=test_type),
            )

            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsNone(validator.target_encoder)
            self.assertIsNone(validator.feature_encoder)

    def test_dataframe_input_categorical(self):
        """
        Makes sure we automatically encode categorical data
        """
        for test_type in ['bool', 'category']:
            validator = InputValidator()
            X = validator.validate_features(
                pd.DataFrame(data=self.X, dtype=test_type),
            )
            y = validator.validate_target(
                pd.DataFrame(data=self.y, dtype=test_type),
                is_classification=True,
            )

            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsNotNone(validator.target_encoder)
            self.assertIsNotNone(validator.feature_encoder)

    def test_binary_conversion(self):
        """
        Makes sure that a encoded target for classification
        properly retains the binary target type
        """
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

        # Make sure binary also works with PD dataframes
        validator = InputValidator()

        # Just 2 classes, 1 and 2
        y_train = validator.validate_target(
            pd.DataFrame([1.0, 2.0, 2.0, 1.0], dtype='category'),
            is_classification=True,
        )
        self.assertEqual('binary', type_of_target(y_train))

    def test_multiclass_conversion(self):
        """
        Makes sure that a encoded target for classification
        properly retains the multiclass target type
        """
        # Multiclass conversion for different datatype
        for input_object in [
            [1.0, 2.0, 2.0, 4.0, 3],
            np.array([1.0, 2.0, 2.0, 4.0, 3], dtype=np.float64),
            pd.DataFrame([1.0, 2.0, 2.0, 4.0, 3], dtype='category'),
        ]:
            validator = InputValidator()
            y_train = validator.validate_target(
                input_object,
                is_classification=True,
            )
            self.assertEqual('multiclass', type_of_target(y_train))

    def test_multilabel_conversion(self):
        """
        Makes sure that a encoded target for classification
        properly retains the multilabel target type
        """
        # Multi-label conversion for different datatype
        for input_object in [
            [[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
            np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
            pd.DataFrame([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype='category'),
        ]:
            validator = InputValidator()
            y_train = validator.validate_target(
                input_object,
                is_classification=True,
            )
            self.assertEqual('multilabel-indicator', type_of_target(y_train))

    def test_continuous_multioutput_conversion(self):
        """
        Makes sure that an input for regression
        properly retains the multiout continious target type
        """
        # Regression multi out conversion for different datatype
        for input_object in [
            [[31.4, 94], [40.5, 109], [25.0, 30]],
            np.array([[31.4, 94], [40.5, 109], [25.0, 30]]),
            pd.DataFrame([[31.4, 94], [40.5, 109], [25.0, 30]]),
        ]:
            validator = InputValidator()
            y_train = validator.validate_target(
                input_object,
                is_classification=False,
            )
            self.assertEqual('continuous-multioutput', type_of_target(y_train))

    def test_regression_conversion(self):
        """
        Makes sure that a regression input
        properly retains the continious target type
        """
        for input_object in [
            [1.0, 76.9, 123, 4.0, 81.1],
            np.array([1.0, 76.9, 123, 4.0, 81.1]),
            pd.DataFrame([1.0, 76.9, 123, 4.0, 81.1]),
        ]:
            validator = InputValidator()
            y_train = validator.validate_target(
                input_object,
                is_classification=False,
            )
            self.assertEqual('continuous', type_of_target(y_train))

    def test_dataframe_input_unsupported(self):
        """
        Makes sure we raise a proper message to the user,
        when providing not supported data input
        """
        validator = InputValidator()
        with self.assertRaisesRegex(ValueError, "Auto-sklearn does not support time"):
            validator.validate_features(
                pd.DataFrame({'datetime': [pd.Timestamp('20180310')]})
            )
        with self.assertRaisesRegex(ValueError, "has invalid type object"):
            validator.validate_features(
                pd.DataFrame({'string': ['foo']})
            )

        validator = InputValidator()
        with self.assertRaisesRegex(ValueError, "Expected 2D array, got"):
            validator.validate_features({'input1': 1, 'input2': 2})

        validator = InputValidator()
        with self.assertRaisesRegex(ValueError, "Expected 2D array, got"):
            validator.validate_features(InputValidator())

        validator = InputValidator()
        X = pd.DataFrame(data=['a', 'b', 'c'], dtype='category')
        with unittest.mock.patch('autosklearn.data.validation.InputValidator._check_and_get_columns_to_encode') as mock_foo:  # noqa E501
            # Mock that all columns are ok. There should be a
            # checker to catch for bugs
            mock_foo.return_value = ([], [])
            with self.assertRaisesRegex(ValueError, 'Failed to convert the input'):
                validator.validate_features(X)

    def test_dataframe_econding_1D(self):
        """
        Test that the encoding/decoding works in 1D
        """
        validator = InputValidator()
        y = validator.validate_target(
            pd.DataFrame(data=self.y, dtype=bool),
            is_classification=True,
        )
        np.testing.assert_array_almost_equal(np.array([0, 1, 0]), y)

        # Result should not change on a multi call
        y = validator.validate_target(pd.DataFrame(data=self.y, dtype=bool))
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
        """
        Test that the encoding/decoding works in 2D
        """
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

    def test_noNaN(self):
        """
        Makes sure that during classification/regression task,
        the transformed data is not corrupted.

        Testing is given without Nan and no sparse data
        """
        # numpy - categorical - classification
        x = np.array(['a', 'b', 'c', 'a', 'b', 'c']).reshape(-1, 1)
        validator = InputValidator()
        with self.assertRaisesRegex(ValueError,
                                    'the only valid dtypes are numerical ones'):
            x_t, y_t = validator.validate(x, np.copy(x), is_classification=True)

        # numpy - categorical - regression
        with self.assertRaisesRegex(ValueError,
                                    'the only valid dtypes are numerical ones'):
            x_t, y_t = validator.validate(x, np.copy(x), is_classification=False)

        # numpy - numerical - classification
        x = np.random.random_sample((4, 4))
        y = np.random.choice([0, 1], 4)
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=True)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.issubdtype(y_t.dtype, np.number))
        self.assertEqual(type_of_target(y_t), 'binary')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))
        self.assertTupleEqual(np.shape(y), np.shape(y_t))

        # numpy - numerical - regression
        x = np.random.random_sample((4, 4))
        y = np.random.random_sample(4)
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=False)
        np.testing.assert_array_equal(x, x_t)  # No change to valid data
        np.testing.assert_array_equal(y, y_t)
        self.assertEqual(type_of_target(y_t), 'continuous')

        # pandas - categorical - classification
        x = pd.DataFrame({'A': np.random.choice(['a', 'b'], 4),
                          'B': np.random.choice(['a', 'b'], 4)},
                         dtype='category')
        y = pd.DataFrame(np.random.choice(['c', 'd'], 4), dtype='category')
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=True)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.issubdtype(y_t.dtype, np.number))
        self.assertEqual(type_of_target(y_t), 'binary')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))
        self.assertTupleEqual(np.shape(y.to_numpy().reshape(-1)), np.shape(y_t))  # ravel

        # pandas - categorical - regression
        x = pd.DataFrame({'A': np.random.choice(['a', 'b'], 4),
                          'B': np.random.choice(['a', 'b'], 4)},
                         dtype='category')
        y = pd.DataFrame(np.random.random_sample(4))
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=False)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.issubdtype(y_t.dtype, np.number))
        self.assertEqual(type_of_target(y_t), 'continuous')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))
        np.testing.assert_array_equal(y.to_numpy().reshape(-1), y_t)
        self.assertTupleEqual(np.shape(y.to_numpy().reshape(-1)), np.shape(y_t))  # ravel version

        # pandas - numerical - classification
        x = pd.DataFrame({'A': np.random.random_sample(4),
                          'B': np.random.choice([2.5, 1.2], 4)})
        y = pd.DataFrame([1.0, 2.2, 3.2, 2.2])
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=True)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.issubdtype(y_t.dtype, np.number))
        self.assertEqual(type_of_target(y_t), 'multiclass')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))
        np.testing.assert_array_equal(np.array([0, 1, 2, 1]), y_t)
        self.assertTupleEqual(np.shape(y.to_numpy().reshape(-1)), np.shape(y_t))  # ravel

        # pandas - numerical - regression
        x = pd.DataFrame({'A': np.random.choice([1.5, 3.6], 4),
                          'B': np.random.choice([2.5, 1.2], 4)})
        y = pd.DataFrame(np.random.random_sample(4))
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=False)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.issubdtype(y_t.dtype, np.number))
        self.assertEqual(type_of_target(y_t), 'continuous')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))
        self.assertTupleEqual(np.shape(y.to_numpy().reshape(-1)), np.shape(y_t))  # ravel
        np.testing.assert_array_equal(y.to_numpy().reshape(-1), y_t)
        return

    def test_NaN(self):
        # numpy - categorical - classification
        # np.nan in categorical array means that the array will be
        # type string, and np.nan will be casted as 'nan'.
        # In turn, 'nan' will be another category
        x = np.array([1, 2, 3, 4, 5.0, np.nan]).reshape(-1, 1)
        y = np.array([1, 2, 3, 4, 5.0, 6.0]).reshape(-1, 1)
        validator = InputValidator()
        x_t, y_t = validator.validate(x, y, is_classification=True)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.issubdtype(y_t.dtype, np.number))
        self.assertTrue(np.isnan(x_t).any())  # Preserve NaN in features
        self.assertEqual(type_of_target(y_t), 'multiclass')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))

        # numpy - categorical - regression
        # nan in target should raise error
        y = np.random.random_sample((6, 1))
        y[1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Target values cannot contain missing/NaN'):
            InputValidator().validate_target(y)

        # numpy - numerical - classification
        # Numerical numpy features should continue without encoding
        # categorical encoding of Nan for the targets is not supported
        x = np.random.random_sample((4, 4))
        x[3] = np.nan
        y = np.random.choice([0.0, 1.0], 4)
        y[1] = np.nan
        x_t = InputValidator().validate_features(x)
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))
        self.assertTrue(np.isnan(x_t).any())
        self.assertEqual(type_of_target(y_t), 'multiclass')
        self.assertTupleEqual(np.shape(x), np.shape(x_t))

        with self.assertRaisesRegex(ValueError, 'Target values cannot contain missing/NaN'):
            InputValidator().validate_target(y, is_classification=True)

        with self.assertRaisesRegex(ValueError, 'Target values cannot contain missing/NaN'):
            InputValidator().validate_target(y, is_classification=False)

        # Make sure we allow NaN in numerical columns
        x_only_numerical = np.random.random_sample(4)
        x[3] = np.nan
        x_only_numerical = pd.DataFrame(data={'A': x_only_numerical, 'B': x_only_numerical*2})
        try:
            InputValidator().validate_features(x_only_numerical)
        except ValueError:
            self.fail("NaN values in numerical columns is allowed")

        # Make sure we do not allow NaN in categorical columns
        x_only_categorical = pd.DataFrame(data=pd.Series([1, 2, pd.NA], dtype="category"))
        with self.assertRaisesRegex(ValueError, 'Categorical features in a dataframe cannot'):
            InputValidator().validate_features(x_only_categorical)

        y = np.random.choice([0.0, 1.0], 4)
        y[1] = np.nan
        y = pd.DataFrame(y)

        with self.assertRaisesRegex(ValueError, 'Target values cannot contain missing/NaN'):
            InputValidator().validate_target(y, is_classification=True)

        with self.assertRaisesRegex(ValueError, 'Target values cannot contain missing/NaN'):
            InputValidator().validate_target(y, is_classification=False)
        return

    def test_no_new_category_after_fit(self):
        # First make sure no problem if no categorical
        x = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        y = pd.DataFrame([1, 2, 3, 4])
        validator = InputValidator()
        validator.validate(x, y, is_classification=True)
        validator.validate_features(x)
        x['A'] = x['A'].apply(lambda x: x*x)
        validator.validate_features(x)

        # Then make sure we catch categorical extra categories
        x = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, dtype='category')
        y = pd.DataFrame([1, 2, 3, 4])
        validator = InputValidator()
        validator.validate(x, y, is_classification=True)
        validator.validate_features(x)
        x['A'] = x['A'].apply(lambda x: x*x)
        with self.assertRaisesRegex(
            ValueError,
            'During fit, the input features contained categorical values'
        ):
            validator.validate_features(x)

        # For label encoder of targets
        with self.assertRaisesRegex(
            ValueError,
            'During fit, the target array contained the categorical'
        ):
            validator.validate_target(pd.DataFrame([1, 2, 5, 4]))

        # For ordinal encoder of targets
        x = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, dtype='category')
        validator = InputValidator()
        validator.validate(x, x, is_classification=True)
        validator.validate_target(pd.DataFrame(
            {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, dtype='category')
        )
        with self.assertRaisesRegex(
            ValueError,
            'During fit, the target array contained the categorical'
        ):
            validator.validate_target(pd.DataFrame(
                {'A': [1, 2, 3, 4], 'B': [5, 9, 7, 8]}, dtype='category')
            )
        return

    def test_big_dataset_encoding(self):
        x, y = sklearn.datasets.fetch_openml(data_id=2, return_X_y=True, as_frame=True)
        validator = InputValidator()

        with self.assertRaisesRegex(
            ValueError,
            'Categorical features in a dataframe cannot contain missing/NaN'
        ):
            x_t, y_t = validator.validate(x, y, is_classification=True)

        # Make sure translation works apart from Nan

        # NaN is not supported in categories, so
        # drop columns with them. Also, do a proof of concept
        # that all nan column is preserved, so that the pipeline deal
        # with it
        x = x.dropna('columns', 'any')
        x.insert(len(x.columns), 'NaNColumn', np.nan, True)
        x_t, y_t = validator.validate(x, y, is_classification=True)
        self.assertTupleEqual(np.shape(x), np.shape(x_t))

        self.assertTrue(np.all(pd.isnull(x_t[:, -1])))

        # Leave columns that are complete NaN
        # The sklearn pipeline will handle that
        self.assertTrue(np.isnan(x_t).any())
        np.testing.assert_array_equal(
            pd.isnull(x.dropna(axis='columns', how='all')),
            pd.isnull(x.dropna(axis='columns', how='any'))
        )

        # make sure everything was encoded to number
        self.assertTrue(np.issubdtype(x_t.dtype, np.number))

        # No change to numerical columns
        np.testing.assert_array_equal(x['carbon'].to_numpy(), x_t[:, 3])

        # Categorical columns are sorted to the beginning
        self.assertEqual(
            validator.feature_types,
            (['categorical'] * 3) + (['numerical'] * 7)
        )
        self.assertEqual(x.iloc[0, 6], 610)
        np.testing.assert_array_equal(x_t[0], [0, 0, 0, 8, 0, 0, 0.7, 610, 0, np.NaN])

        return

    def test_join_and_check(self):
        validator = InputValidator()

        # Numpy Testing
        y = np.array([2, 2, 3, 4, 5])
        y_test = np.array([3, 4, 5, 6, 1])

        joined = validator.join_and_check(y, y_test)
        np.testing.assert_array_equal(
            joined,
            np.array([2, 2, 3, 4, 5, 3, 4, 5, 6, 1])
        )

        validator.validate_target(joined, is_classification=True)
        y_encoded = validator.validate_target(y)
        y_test_encoded = validator.validate_target(y_test)

        # If a common encoding happened, then common elements
        # should have a common encoding
        self.assertEqual(y_encoded[2], y_test_encoded[0])

        # Pandas Testing
        validator = InputValidator()
        joined = validator.join_and_check(
            pd.DataFrame(y),
            pd.DataFrame(y_test)
        )
        np.testing.assert_array_equal(
            joined,
            pd.DataFrame([2, 2, 3, 4, 5, 3, 4, 5, 6, 1])
        )

        # List Testing
        validator = InputValidator()
        joined = validator.join_and_check(
            [2, 2, 3, 4, 5],
            [3, 4, 5, 6, 1]
        )
        np.testing.assert_array_equal(
            joined,
            [2, 2, 3, 4, 5, 3, 4, 5, 6, 1]
        )

        # Make sure some messages are triggered
        y = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
        y_test = np.array([3, 4, 5, 6, 1])
        with self.assertRaisesRegex(
            ValueError,
            'Train and test targets must have the same dimensionality'
        ):
            joined = validator.join_and_check(y, y_test)
        with self.assertRaisesRegex(
            ValueError,
            'Train and test targets must be of the same type'
        ):
            joined = validator.join_and_check(y, pd.DataFrame(y_test))

    def test_big_dataset_encoding2(self):
        """
        Makes sure that when there are multiple classes,
        and test/train targets differ, we proactively encode together
        the data between test and train
        """
        X, y = sklearn.datasets.fetch_openml(data_id=183, return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X,
            y,
            random_state=1
        )

        # Make sure this test makes sense, so that y_test
        # and y_train have different classes
        all_classes = set(np.unique(y_test)).union(set(np.unique(y_train)))
        elements_in_test_only = np.setdiff1d(np.unique(y_test), np.unique(y_train))
        self.assertGreater(len(elements_in_test_only), 0)

        validator = InputValidator()
        common = validator.join_and_check(
            pd.DataFrame(y),
            pd.DataFrame(y_test)
        )

        validator.validate_target(common, is_classification=True)

        encoded_classes = validator.target_encoder.classes_
        missing = all_classes - set(encoded_classes)
        self.assertEqual(len(missing), 0)

    def test_all_posible_dtype_changes(self):
        """We do not allow a change in dtype once inputvalidator
        is fitted"""
        data = [[1, 0, 1], [1, 1, 1]]
        type_perms = list(itertools.permutations([
            data,
            np.array(data),
            pd.DataFrame(data)
        ], r=2))

        for first, second in type_perms:
            validator = InputValidator()
            validator.validate_target(first)
            with self.assertRaisesRegex(ValueError,
                                        "Auto-sklearn previously received targets of type"):
                validator.validate_target(second)
            validator.validate_features(first)
            with self.assertRaisesRegex(ValueError,
                                        "Auto-sklearn previously received features of type"):
                validator.validate_features(second)
