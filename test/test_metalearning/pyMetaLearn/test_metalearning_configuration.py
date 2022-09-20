import logging
import os

import autosklearn.metalearning.optimizers.metalearn_optimizer.metalearner as metalearner  # noqa: E501
import autosklearn.pipeline.classification
from autosklearn.metalearning.metalearning.meta_base import MetaBase

import unittest

logging.basicConfig()


class MetalearningConfiguration(unittest.TestCase):
    def test_metalearning_cs_size(self):
        self.cwd = os.getcwd()
        data_dir = os.path.dirname(__file__)
        data_dir = os.path.join(data_dir, "test_meta_base_data")
        os.chdir(data_dir)

        # Total: 176, categorical: 3, numerical: 7, string: 7
        total = 179
        num_numerical = 6
        num_string = 11
        num_categorical = 3
        for feat_type, cs_size in [
            ({"A": "numerical"}, total - num_string - num_categorical),
            ({"A": "categorical"}, total - num_string - num_numerical),
            ({"A": "string"}, total - num_categorical - num_numerical),
            ({"A": "numerical", "B": "categorical"}, total - num_string),
            ({"A": "numerical", "B": "string"}, total - num_categorical),
            ({"A": "categorical", "B": "string"}, total - num_numerical),
            ({"A": "categorical", "B": "string", "C": "numerical"}, total),
        ]:
            pipeline = autosklearn.pipeline.classification.SimpleClassificationPipeline(
                feat_type=feat_type
            )
            self.cs = pipeline.get_hyperparameter_search_space(feat_type=feat_type)

            self.logger = logging.getLogger()
            meta_base = MetaBase(self.cs, data_dir, logger=self.logger)
            self.meta_optimizer = metalearner.MetaLearningOptimizer(
                "233", self.cs, meta_base, logger=self.logger
            )
            self.assertEqual(
                len(self.meta_optimizer.configuration_space), cs_size, feat_type
            )
