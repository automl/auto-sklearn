import unittest

import numpy as np

import scipy.sparse
import openml
import sklearn.tree
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from autosklearn.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder
from autosklearn.pipeline.implementations.CategoryShift import CategoryShift

sparse1 = scipy.sparse.csc_matrix(([3, 2, 1, 1, 2, 3],
                                   ((1, 4, 5, 2, 3, 5),
                                    (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse1_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 1, 1],
                                      ((5, 4, 1, 2, 3, 5),
                                       (0, 1, 2, 3, 4, 5))), shape=(6, 6))

sparse2 = scipy.sparse.csc_matrix(([2, 1, 0, 0, 0, 0],
                                   ((1, 4, 5, 2, 3, 5),
                                    (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse2_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 1, 1],
                                      ((5, 4, 1, 2, 3, 5),
                                       (0, 1, 2, 3, 3, 3))), shape=(6, 4))

sparse2_csr = scipy.sparse.csr_matrix(([2, 1, 0, 0, 0, 0],
                                      ((1, 4, 5, 2, 3, 5),
                                       (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse2_csr_1h = scipy.sparse.csr_matrix(([1, 1, 1, 1, 1, 1],
                                         ((5, 4, 1, 2, 3, 5),
                                          (0, 1, 2, 3, 3, 3))), shape=(6, 4))


class TestSparseOneHotEncoder(unittest.TestCase):
    def test_sparse1(self):
        self._fit_then_transform(sparse1_1h.todense(), sparse1)

    def test_sparse2(self):
        self._fit_then_transform(sparse2_1h.todense(), sparse2)

    def test_sparse2_csr(self):
        self._fit_then_transform(sparse2_csr_1h.todense(), sparse2_csr)

    def _fit_then_transform(self, expected, input):
        # Test fit_transform
        input_copy = input.copy()
        ohe = SparseOneHotEncoder()
        transformation = ohe.fit_transform(input)
        self.assertIsInstance(transformation, scipy.sparse.csr_matrix)
        np.testing.assert_array_almost_equal(
            expected.astype(float),
            transformation.todense()
        )
        self._check_arrays_equal(input, input_copy)

        # Test fit, and afterwards transform
        ohe2 = SparseOneHotEncoder()
        ohe2.fit(input)
        transformation = ohe2.transform(input)
        self.assertIsInstance(transformation, scipy.sparse.csr_matrix)
        np.testing.assert_array_almost_equal(expected, transformation.todense())
        self._check_arrays_equal(input, input_copy)

    def _check_arrays_equal(self, a1, a2):
        if scipy.sparse.issparse(a1):
            a1 = a1.toarray()
        if scipy.sparse.issparse(a2):
            a2 = a2.toarray()
        np.testing.assert_array_almost_equal(a1, a2)

    def test_transform_with_unknown_value(self):
        # fit_data: this is going to be used to fit.
        # note that 0 is no category because the data here is sparse.
        fit_data = np.array(((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))).transpose()
        fds = scipy.sparse.csr_matrix(fit_data)
        ohe = SparseOneHotEncoder()
        ohe.fit(fds)
        # transf_data: this is going to be used in a transform call.
        # Note that transf_data has categories not seen at the fit.
        # Unseen categories are ignored (encoded just with zeros).
        transf_data = np.array(((0, 1, 2, 6), (0, 1, 6, 7))).transpose()
        tds = scipy.sparse.csr_matrix(transf_data)
        output = ohe.transform(tds).todense()
        # From tds, just 3 categories (1 and 2 in the 1st feature and 1 in the 2nd
        # feature) have been seen during fit, therefore:
        self.assertEqual(3, np.sum(output))

    def test_classification_workflow(self):
        task = openml.tasks.get_task(254)
        X, y = task.get_X_and_y()

        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=3,
                                                     train_size=0.5,
                                                     test_size=0.5)

        X_train = scipy.sparse.csc_matrix(X_train)
        X_test = scipy.sparse.csc_matrix(X_test)

        pipeline = sklearn.pipeline.Pipeline((
            ('shift', CategoryShift()),
            ('imput', SimpleImputer(strategy='constant', fill_value=2)),
            ('ohe', SparseOneHotEncoder()),
            ('tree', DecisionTreeClassifier(random_state=1)),
            ))

        pipeline.fit(X_train, y_train)
        pred_train = pipeline.predict(X_train)
        self.assertTrue((pred_train == y_train).all())
        # With an incorrect copy operation the OneHotEncoder would rearrange
        # the data in such a way that the accuracy would drop to 66%
        pred_test = pipeline.predict(X_test)
        self.assertTrue((pred_test == y_test).all())
