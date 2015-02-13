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


class MetafeatureValueDummy(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value


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

        # TODO accept float/ints instead of string in the HPOlibConfigSpace
        configuration_space.get_hyperparameter(
            'gradient_boosting:n_estimators').choices = [100]
        configuration_space.get_hyperparameter(
            'random_forest:n_estimators').choices = [100]
        configuration_space.get_hyperparameter(
            'extra_trees:n_estimators').choices = [100]
        configuration_space.get_hyperparameter(
            'k_nearest_neighbors:leaf_size').choices = [30]
        configuration_space.get_hyperparameter(
            'k_nearest_neighbors:p').choices = [1, 2, 5]
        configuration_space.get_hyperparameter(
            'libsvm_svc:max_iter').choices = [-1]
        configuration_space.get_hyperparameter(
            'liblinear:intercept_scaling').choices = [1]

        rs = RandomSampler(configuration_space, 1)
        mock_mlo.return_value = [Run(rs.sample_configuration(), 1, 1)]

        ml = MetaLearning()
        ml._metafeatures_encoded_labels = DatasetMetafeatures("1",
            [MetafeatureValueDummy("a", 1), MetafeatureValueDummy("b", 1)])
        ml._metafeatures_labels = DatasetMetafeatures("2",
            [MetafeatureValueDummy("c", 1),
             MetafeatureValueDummy("d", 1),
             MetafeatureValueDummy("e", 1)])
        initial_configuration_strings_for_smac = \
            ml.create_metalearning_string_for_smac_call(
            configuration_space, "iris", "bac_metric")

        self.assertEqual(["--initial-challengers "
                          "-rescaling:strategy 'min/max' "
                          "-gradient_boosting:min_samples_leaf '8' "
                          "-imputation:strategy 'most_frequent' "
                          "-gradient_boosting:subsample '0.545998348066' "
                          "-gradient_boosting:max_depth '1' "
                          "-preprocessor 'None' "
                          "-gradient_boosting:min_samples_split '10' "
                          "-gradient_boosting:learning_rate "
                          "'0.000822335208483' "
                          "-gradient_boosting:n_estimators '100' "
                          "-gradient_boosting:max_features '4.55642355925' "
                          "-classifier 'gradient_boosting'"],
                         initial_configuration_strings_for_smac)