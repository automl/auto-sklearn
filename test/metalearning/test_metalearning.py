import unittest
from ParamSklearn.util import get_dataset

from autosklearn.metalearning.metalearning import MetaLearning
from autosklearn.models.paramsklearn import get_configuration_space


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

    def test_metalearning(self):
        dataset_name = 'digits'

        initial_challengers = {'acc_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'none' "
                                    "-classifier 'random_forest' "
                                    "-imputation:strategy 'mean' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-random_forest:bootstrap 'True' "
                                    "-random_forest:criterion 'gini' "
                                    "-random_forest:max_depth 'None' "
                                    "-random_forest:max_features '1.0' "
                                    "-random_forest:max_leaf_nodes 'None' "
                                    "-random_forest:min_samples_leaf '1' "
                                    "-random_forest:min_samples_split '2' "
                                    "-random_forest:n_estimators '100' "
                                    "-rescaling:strategy 'min/max'\""
                                   ],
                               'auc_metric':
                                    ["--initial-challengers \" "
                                     "-balancing:strategy 'weighting' "
                                     "-classifier 'gradient_boosting' "
                                     "-gradient_boosting:learning_rate '0.00438932235819' "
                                     "-gradient_boosting:max_depth '8' "
                                     "-gradient_boosting:max_features '2.47034566014' "
                                     "-gradient_boosting:min_samples_leaf '3' "
                                     "-gradient_boosting:min_samples_split '6' "
                                     "-gradient_boosting:n_estimators '100' "
                                     "-gradient_boosting:subsample '0.601991494509' "
                                     "-imputation:strategy 'mean' "
                                     "-preprocessor 'no_preprocessing' "
                                     "-rescaling:strategy 'none'\""
                                    ],
                               'bac_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'weighting' "
                                    "-classifier 'gradient_boosting' "
                                    "-gradient_boosting:learning_rate '0.00438932235819' "
                                    "-gradient_boosting:max_depth '8' "
                                    "-gradient_boosting:max_features '2.47034566014' "
                                    "-gradient_boosting:min_samples_leaf '3' "
                                    "-gradient_boosting:min_samples_split '6' "
                                    "-gradient_boosting:n_estimators '100' "
                                    "-gradient_boosting:subsample '0.601991494509' "
                                    "-imputation:strategy 'mean' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-rescaling:strategy 'none'\""
                                   ],
                               'f1_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'none' "
                                    "-classifier 'random_forest' "
                                    "-imputation:strategy 'mean' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-random_forest:bootstrap 'True' "
                                    "-random_forest:criterion 'gini' "
                                    "-random_forest:max_depth 'None' "
                                    "-random_forest:max_features '1.0' "
                                    "-random_forest:max_leaf_nodes 'None' "
                                    "-random_forest:min_samples_leaf '1' "
                                    "-random_forest:min_samples_split '2' "
                                    "-random_forest:n_estimators '100' "
                                    "-rescaling:strategy 'min/max'\""
                                   ],
                               'pac_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'none' "
                                    "-classifier 'random_forest' "
                                    "-imputation:strategy 'mean' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-random_forest:bootstrap 'True' "
                                    "-random_forest:criterion 'gini' "
                                    "-random_forest:max_depth 'None' "
                                    "-random_forest:max_features '1.0' "
                                    "-random_forest:max_leaf_nodes 'None' "
                                    "-random_forest:min_samples_leaf '1' "
                                    "-random_forest:min_samples_split '2' "
                                    "-random_forest:n_estimators '100' "
                                    "-rescaling:strategy 'min/max'\""

                                   ]}

        for metric in initial_challengers:
            configuration_space = get_configuration_space(
                {'metric': metric,
                 'task': 'multiclass.classification',
                 'is_sparse': False}, include_preprocessors=['no_preprocessing'])

            X_train, Y_train, X_test, Y_test = get_dataset(dataset_name)
            categorical = [False] * X_train.shape[1]

            ml = MetaLearning()
            ml.calculate_metafeatures_with_labels(
                    X_train, Y_train, categorical, dataset_name)
            ml.calculate_metafeatures_encoded_labels(
                    X_train, Y_train, categorical, dataset_name)
            initial_configuration_strings_for_smac = \
                ml.create_metalearning_string_for_smac_call(
                    configuration_space, dataset_name, metric,
                    'multiclass.classification', False, 1, None)

            print metric
            self.assertEqual(initial_challengers[metric],
                             initial_configuration_strings_for_smac)