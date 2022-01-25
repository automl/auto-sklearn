import logging
import os
import unittest

import pandas as pd

import autosklearn.pipeline.classification
from autosklearn.metalearning.metalearning.meta_base import MetaBase


class MetaBaseTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.cwd = os.getcwd()
        data_dir = os.path.dirname(__file__)
        data_dir = os.path.join(data_dir, 'test_meta_base_data')
        os.chdir(data_dir)

        cs = autosklearn.pipeline.classification.SimpleClassificationPipeline()\
            .get_hyperparameter_search_space()

        self.logger = logging.getLogger()
        self.base = MetaBase(cs, data_dir, logger=self.logger)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_get_all_runs(self):
        runs = self.base.get_all_runs()
        self.assertIsInstance(runs, pd.DataFrame)
        # TODO update this ASAP
        self.assertEqual((125, 125), runs.shape)

    def test_get_runs(self):
        runs = self.base.get_runs('233')
        # TODO update this ASAP
        self.assertEqual(125, len(runs))
        self.assertIsInstance(runs, pd.Series)

    def test_get_metafeatures_single_dataset(self):
        mf = self.base.get_metafeatures('233')
        self.assertIsInstance(mf, pd.Series)
        self.assertEqual(mf.name, '233')
        self.assertEqual(mf.loc['NumberOfInstances'], 2142.0)

    def test_get_metafeatures_single_feature(self):
        mf = self.base.get_metafeatures(features='NumberOfInstances')
        self.assertIsInstance(mf, pd.Series)
        self.assertEqual(mf.shape, (132, ))

    def test_get_metafeatures_single_dataset_and_single_feature(self):
        mf = self.base.get_metafeatures('233', features='NumberOfInstances')
        self.assertEqual(mf.shape, ())

    def test_get_metafeatures_multiple_datasets(self):
        mf = self.base.get_metafeatures(['233', '236'])
        self.assertIsInstance(mf, pd.DataFrame)
        self.assertEqual(mf.shape, (2, 46))

    def test_get_metafeatures_multiple_features(self):
        mf = self.base.get_metafeatures(features=['NumberOfInstances',
                                                  'NumberOfClasses'])
        self.assertIsInstance(mf, pd.DataFrame)
        self.assertEqual(mf.shape, (132, 2))
