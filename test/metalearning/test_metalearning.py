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
                                    "-balancing:strategy 'weighting' "
                                    "-classifier 'random_forest' "
                                    "-imputation:strategy 'most_frequent' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-random_forest:bootstrap 'True' "
                                    "-random_forest:criterion 'gini' "
                                    "-random_forest:max_depth 'None' "
                                    "-random_forest:max_features '1.66027054533' "
                                    "-random_forest:max_leaf_nodes 'None' "
                                    "-random_forest:min_samples_leaf '2' "
                                    "-random_forest:min_samples_split '20' "
                                    "-random_forest:n_estimators '100' "
                                    "-rescaling:strategy 'min/max'\""],
                               'auc_metric':
                                    ["--initial-challengers \" "
                                     "-balancing:strategy 'weighting' "
                                     "-classifier 'random_forest' "
                                     "-imputation:strategy 'most_frequent' "
                                     "-preprocessor 'no_preprocessing' "
                                     "-random_forest:bootstrap 'True' "
                                     "-random_forest:criterion 'gini' "
                                     "-random_forest:max_depth 'None' "
                                     "-random_forest:max_features '1.66027054533' "
                                     "-random_forest:max_leaf_nodes 'None' "
                                     "-random_forest:min_samples_leaf '2' "
                                     "-random_forest:min_samples_split '20' "
                                     "-random_forest:n_estimators '100' "
                                     "-rescaling:strategy 'min/max'\""
                                    ],
                               'bac_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'weighting' "
                                    "-classifier 'libsvm_svc' "
                                    "-imputation:strategy 'median' "
                                    "-libsvm_svc:C '5486.70315669' "
                                    "-libsvm_svc:class_weight 'auto' "
                                    "-libsvm_svc:gamma '0.29783505435' "
                                    "-libsvm_svc:kernel 'rbf' "
                                    "-libsvm_svc:max_iter '-1' "
                                    "-libsvm_svc:shrinking 'True' "
                                    "-libsvm_svc:tol '0.0962993142836' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-rescaling:strategy 'min/max'\""
                                    ],
                               'f1_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'weighting' "
                                    "-classifier 'random_forest' "
                                    "-imputation:strategy 'median' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-random_forest:bootstrap 'False' "
                                    "-random_forest:criterion 'gini' "
                                    "-random_forest:max_depth 'None' "
                                    "-random_forest:max_features '1.04361822759' "
                                    "-random_forest:max_leaf_nodes 'None' "
                                    "-random_forest:min_samples_leaf '3' "
                                    "-random_forest:min_samples_split '6' "
                                    "-random_forest:n_estimators '100' "
                                    "-rescaling:strategy 'min/max'\""],
                               'pac_metric':
                                   ["--initial-challengers \" "
                                    "-balancing:strategy 'weighting' "
                                    "-classifier 'random_forest' "
                                    "-imputation:strategy 'most_frequent' "
                                    "-preprocessor 'no_preprocessing' "
                                    "-random_forest:bootstrap 'True' "
                                    "-random_forest:criterion 'gini' "
                                    "-random_forest:max_depth 'None' "
                                    "-random_forest:max_features '1.66027054533' "
                                    "-random_forest:max_leaf_nodes 'None' "
                                    "-random_forest:min_samples_leaf '2' "
                                    "-random_forest:min_samples_split '20' "
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
                    configuration_space, dataset_name, metric, 1)

            print metric
            self.assertEqual(initial_challengers[metric],
                             initial_configuration_strings_for_smac)