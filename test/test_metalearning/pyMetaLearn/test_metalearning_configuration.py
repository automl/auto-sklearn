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

        for feat_type, cs_size in [
            ({"A": "numerical"}, 165),
            ({"A": "categorical"}, 162),
            ({"A": "string"}, 174),
            ({"A": "numerical", "B": "categorical"}, 168),
            ({"A": "numerical", "B": "string"}, 180),
            ({"A": "categorical", "B": "string"}, 177),
            ({"A": "categorical", "B": "string", "C": "numerical"}, 183),
        ]:
            pipeline = autosklearn.pipeline.classification.SimpleClassificationPipeline(
                feat_type=feat_type
            )
            self.cs = pipeline.get_hyperparameter_search_space(feat_type=feat_type)
            # print(self.cs.get_default_configuration())

            self.logger = logging.getLogger()
            meta_base = MetaBase(self.cs, data_dir, logger=self.logger)
            self.meta_optimizer = metalearner.MetaLearningOptimizer(
                "233", self.cs, meta_base, logger=self.logger
            )
            self.assertEqual(len(self.meta_optimizer.configuration_space), cs_size)
