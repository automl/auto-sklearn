import os
import unittest

import numpy as np
import pandas as pd

import autosklearn.pipeline.classification
from autosklearn.metalearning.metalearning.meta_base import MetaBase, Run


class MetaBaseTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.cwd = os.getcwd()
        data_dir = os.path.dirname(__file__)
        data_dir = os.path.join(data_dir, 'test_meta_base_data')
        os.chdir(data_dir)

        cs = autosklearn.pipeline.classification.SimpleClassificationPipeline()\
            .get_hyperparameter_search_space()

        self.base = MetaBase(cs, data_dir)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_get_all_runs(self):
        runs = self.base.get_all_runs()
        self.assertIsInstance(runs, pd.DataFrame)
        # TODO update this ASAP
        self.assertEqual((134, 24), runs.shape)

    def test_get_runs(self):
        runs = self.base.get_runs('38_acc')
        # TODO update this ASAP
        self.assertEqual(24, len(runs))
        self.assertIsInstance(runs, pd.Series)

    def test_get_metafeatures_single_dataset(self):
        mf = self.base.get_metafeatures('38_acc')
        self.assertIsInstance(mf, pd.Series)
        self.assertEqual(mf.name, u'38_acc')
        self.assertEqual(mf.loc['NumberOfInstances'], 2527.0)

    def test_get_metafeatures_single_feature(self):
        mf = self.base.get_metafeatures(features='NumberOfInstances')
        self.assertIsInstance(mf, pd.Series)
        self.assertEqual(mf.shape, (140, ))

    def test_get_metafeatures_single_dataset_and_single_feature(self):
        mf = self.base.get_metafeatures('38_acc', features='NumberOfInstances')
        self.assertEqual(mf.shape, ())

    def test_get_metafeatures_multiple_datasets(self):
        mf = self.base.get_metafeatures(['38_acc', '24_acc'])
        self.assertIsInstance(mf, pd.DataFrame)
        self.assertEqual(mf.shape, (2, 46))

    def test_get_metafeatures_multiple_features(self):
        mf = self.base.get_metafeatures(features=['NumberOfInstances',
                                                  'NumberOfClasses'])
        self.assertIsInstance(mf, pd.DataFrame)
        self.assertEqual(mf.shape, (140, 2))

    def test_remove_dataset(self):
        name = "1000_acc"
        for key in self.base.algorithm_runs:
            self.assertIn(name, self.base.algorithm_runs[key].index)
        self.assertIn(name, self.base.metafeatures.index)
        metafeatures_shape = self.base.metafeatures.shape
        self.base.remove_dataset(name)
        for key in self.base.algorithm_runs:
            self.assertNotIn(name, self.base.algorithm_runs[key].index)
        self.assertNotIn(name, self.base.metafeatures.index)
        # Check that only one thing was removed
        self.assertEqual(self.base.metafeatures.shape,
                         (metafeatures_shape[0] - 1, metafeatures_shape[1]))

