# -*- encoding: utf-8 -*-
import unittest

import numpy as np

import autosklearn
import ParamSklearn.util as putil


class EstimatorTest(unittest.TestCase):

    def test_fit_OneHotEncoder(self):
        # Test if the OneHotEncoder is called
        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        X_train = np.hstack((X_train,
                             np.arange(X_train.shape[0]).reshape((-1, 1))))
        cls = autosklearn.AutoSklearnClassifier(time_left_for_this_task=5,
                                                per_run_time_limit=5)
        cls.fit(X_train, Y_train,
                feat_type=['NUMERICAL', 'NUMERICAL', 'NUMERICAL', 'NUMERICAL',
                           'CATEGORICAL'])
        self.assertEqual([False, False, False, False, True],
                         cls.ohe_.categorical_features)
