import logging
import numpy as np
import os
import unittest

import pandas as pd

from ConfigSpace.configuration_space import Configuration
import autosklearn.pipeline.classification

import autosklearn.metalearning.optimizers.metalearn_optimizer.metalearner as metalearner
from autosklearn.metalearning.metalearning.meta_base import MetaBase

logging.basicConfig()


class MetaLearnerTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.cwd = os.getcwd()
        data_dir = os.path.dirname(__file__)
        data_dir = os.path.join(data_dir, 'test_meta_base_data')
        os.chdir(data_dir)

        self.cs = autosklearn.pipeline.classification\
            .SimpleClassificationPipeline().get_hyperparameter_search_space()

        self.logger = logging.getLogger()
        meta_base = MetaBase(self.cs, data_dir, logger=self.logger)
        self.meta_optimizer = metalearner.MetaLearningOptimizer(
            '233', self.cs, meta_base, logger=self.logger)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_metalearning_suggest_all(self):
        ret = self.meta_optimizer.metalearning_suggest_all()
        self.assertEqual(124, len(ret))
        # Reduced to 17 as we changed QDA searchspace
        self.assertEqual('gradient_boosting', ret[0]['classifier:__choice__'])
        self.assertEqual('adaboost', ret[1]['classifier:__choice__'])
        # There is no test for exclude_double_configuration as it's not present
        # in the test data

    def test_metalearning_suggest_all_nan_metafeatures(self):
        self.meta_optimizer.meta_base.metafeatures.loc["233"].iloc[:10] = np.NaN
        ret = self.meta_optimizer.metalearning_suggest_all()
        self.assertEqual(124, len(ret))
        # Reduced to 17 as we changed QDA searchspace
        self.assertEqual('gradient_boosting', ret[0]['classifier:__choice__'])
        self.assertEqual('gradient_boosting', ret[1]['classifier:__choice__'])

    def test_metalearning_suggest(self):
        ret = self.meta_optimizer.metalearning_suggest([])
        self.assertIsInstance(ret, Configuration)
        self.assertEqual('gradient_boosting', ret['classifier:__choice__'])

        ret2 = self.meta_optimizer.metalearning_suggest([ret])
        self.assertIsInstance(ret2, Configuration)
        self.assertEqual('adaboost', ret2['classifier:__choice__'])

    def test_learn(self):
        # Test only some special cases which are probably not yet handled
        # like the metafeatures to eliminate and the random forest
        # hyperparameters
        self.meta_optimizer._learn()

    def test_split_metafeature_array(self):
        ds_metafeatures, other_metafeatures = self.meta_optimizer. \
            _split_metafeature_array()
        self.assertIsInstance(ds_metafeatures, pd.Series)
        self.assertEqual(ds_metafeatures.shape, (46,))
        self.assertIsInstance(other_metafeatures, pd.DataFrame)
        self.assertEqual(other_metafeatures.shape, (131, 46))


if __name__ == "__main__":
    unittest.main()
