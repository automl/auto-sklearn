import mock
import os
import unittest
from AutoSklearn.util import get_dataset
from HPOlibConfigSpace.converters import pcs_parser
from HPOlibConfigSpace.random_sampler import RandomSampler

from AutoML2015.metalearning.metalearning import MetaLearning
from AutoML2015.metalearning import metalearning
from pyMetaLearn.metafeatures.metafeature import DatasetMetafeatures
from pyMetaLearn.optimizers.metalearn_optimizer.metalearner import MetaLearningOptimizer
from pyMetaLearn.metalearning.meta_base import Run


class Test(unittest.TestCase):
    def setUp(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = \
            get_dataset('iris')

        eliminate_class_two = self.Y_train != 2
        self.X_train = self.X_train[eliminate_class_two]
        self.Y_train = self.Y_train[eliminate_class_two]

    @mock.patch.object(MetaLearningOptimizer, "metalearning_suggest_all", autospec=True)
    def test_metalearning(self, mock_mlo):
        with open(os.path.join(os.path.dirname(metalearning.__file__),
                               "files", "params.pcs")) as fh:
            configuration_space = pcs_parser.read(fh)

        rs = RandomSampler(configuration_space, 1)
        mock_mlo.return_value = [Run(rs.sample_configuration(), 1, 1)]

        ml = MetaLearning()
        ml._metafeatures_encoded_labels = DatasetMetafeatures("1", ["a", "b"])
        ml._metafeatures_labels = DatasetMetafeatures("2", ["c", "d", "e"])
        initial_configuration_strings_for_smac = \
            ml.create_metalearning_string_for_smac_call(
            configuration_space, "iris", "bac_metric")

        self.assertEqual(['--initial_challengers -rescaling:strategy min/max'
                          ' -random_forest:min_samples_split 14 '
                          '-random_forest:max_leaf_nodes None '
                          '-random_forest:criterion gini '
                          '-random_forest:max_features 4.26910089048 '
                          '-random_forest:min_samples_leaf 1 '
                          '-imputation:strategy mean -preprocessor None '
                          '-random_forest:max_depth None '
                          '-random_forest:n_estimators 100 '
                          '-random_forest:bootstrap True '
                          '-classifier random_forest'],
                         initial_configuration_strings_for_smac)