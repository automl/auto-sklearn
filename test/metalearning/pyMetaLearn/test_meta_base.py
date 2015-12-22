from __future__ import print_function
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

        cs = autosklearn.pipeline.classification.SimpleClassificationPipeline\
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

    def test_get_metafeatures_as_pandas(self):
        mf = self.base.get_metafeatures('38_acc')
        self.assertTrue(np.isfinite(mf).all())
        self.assertEqual(type(mf), pd.Series)
        self.assertEqual(mf.name, u'38_acc')
        self.assertEqual(mf.loc['NumberOfInstances'], 2527.0)

    def test_get_all_metafeatures_as_pandas(self):
        mf = self.base.get_all_metafeatures()
        self.assertIsInstance(mf, pd.DataFrame)
        self.assertEqual((140, 46), mf.shape)

