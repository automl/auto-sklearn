from __future__ import print_function
import os
import unittest

import numpy as np

import autosklearn.data.arff_data_manager as arff_data_manager
from autosklearn.constants import *


class ArffDataManagerTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.path_to_arff = os.path.join(
            os.path.dirname(__file__), '..', '.data', 'germancredit')
        self.path_to_train_arff = os.path.join(self.path_to_arff, "train.arff")

    def test_load_arff(self):
        X, y, dataset_name, feat_type = arff_data_manager._load_arff(
            self.path_to_train_arff, 'class')

        self.assertEqual(X.shape, (700, 20))
        self.assertEqual(y.shape, (700,))
        self.assertEqual(dataset_name, 'german_credit-weka.filters.unsupervised.instance.Resample-S0-Z30.0-no-replacement-V')
        self.assertEqual(feat_type, ['Categorical', 'Numerical',
                                     'Categorical', 'Categorical',
                                     'Numerical', 'Categorical',
                                     'Categorical', 'Numerical',
                                     'Categorical', 'Categorical',
                                     'Numerical', 'Categorical',
                                     'Numerical', 'Categorical',
                                     'Categorical', 'Numerical',
                                     'Categorical', 'Numerical',
                                     'Categorical', 'Categorical'])
        self.assertEqual(X.dtype, np.float)
        self.assertEqual(y.dtype, np.float)

    def test_ARFFDataManager(self):
        adm = arff_data_manager.ARFFDataManager(self.path_to_arff,
                                               'binary.classification',
                                               'acc_metric', 'class', True)
        self.assertIsInstance(adm.data['X_train'], np.ndarray)
        self.assertIsInstance(adm.data['Y_train'], np.ndarray)
        self.assertIsInstance(adm.data['X_test'], np.ndarray)
        self.assertIsInstance(adm.data['Y_test'], np.ndarray)
        self.assertEqual(adm.info['task'], BINARY_CLASSIFICATION)
        self.assertEqual(adm.info['metric'], ACC_METRIC)
        self.assertEqual(adm.info['is_sparse'], 0)
        self.assertFalse(adm.info['has_missing'])

