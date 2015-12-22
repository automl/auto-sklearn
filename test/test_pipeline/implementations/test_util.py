import unittest

import numpy as np
from sklearn.utils.testing import assert_array_almost_equal

from autosklearn.pipeline.implementations.util import softmax

class UtilTest(unittest.TestCase):
    def test_softmax_binary(self):
        df = np.array([-40.00643897, 34.69754581, 23.71181359 -29.89724287,
                       27.06071791, -37.78334103, -40.15812461, 40.16139229,
                       -27.85887801, 42.67404756, -36.89753589 -36.45148009,
                       54.68976306, 19.47886562, -49.99821027, -35.70205302,
                       -40.59639267, 32.96343916, -39.23777841, -37.86535019,
                       -33.10196906, 26.84144377, -36.8569686])
        probas = softmax(df)
        expected = [[1., 0.], [0., 1.], [0.99794501, 0.00205499],
                    [0., 1.], [1., 0.], [1., 0.], [0., 1.],
                    [1., 0.], [0., 1.], [1., 0.], [0., 1.],
                    [0., 1.], [1., 0.], [1., 0.], [1., 0.],
                    [0., 1.], [1., 0.], [1., 0.], [1., 0.],
                    [0., 1.], [1., 0.]]
        assert_array_almost_equal(expected, probas)

    def test_softmax(self):
        df = np.array([[2.75021367e+10, -8.83772371e-01, -2.20516715e+27],
                       [-2.10848072e+11, 2.35024444e-01, 5.20106536e+25]])
        # With a numerically unstable softmax, the output would be something
        # like this:
        # [[  0.   0.  nan]
        # [nan   0.   0.]]
        probas = softmax(df)
        expected = np.array([[1, 0, 0], [0, 0, 1]])
        self.assertTrue((expected == probas).all())

        df = np.array([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
        probas = softmax(df)
        expected = np.array([[0.25838965, 0.42601251, 0.31559783],
                             [0.28943311, 0.31987306, 0.39069383]])
        assert_array_almost_equal(expected, probas)


