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

        for metric in ['acc_metric', 'auc_metric', 'bac_metric', 'f1_metric',
                       'pac_metric']:
            print metric
            configuration_space = get_configuration_space(
                {'metric': metric,
                 'task': 'multiclass.classification',
                 'is_sparse': False})

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

            self.assertEqual(["--initial-challengers \" "
                              "-classifier 'libsvm_svc' "
                              "-imputation:strategy 'mean' "
                              "-libsvm_svc:C '55.4450303414' "
                              "-libsvm_svc:class_weight 'auto' "
                              "-libsvm_svc:gamma '0.333079079137' "
                              "-libsvm_svc:kernel 'rbf' "
                              "-libsvm_svc:max_iter '-1' "
                              "-libsvm_svc:shrinking 'False' "
                              "-libsvm_svc:tol '0.000167717946595' "
                              "-preprocessor 'no_preprocessing' "
                              "-rescaling:strategy 'min/max'\""],
                             initial_configuration_strings_for_smac)