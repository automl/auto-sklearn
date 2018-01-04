import sklearn.metrics
import sklearn.svm

from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
from autosklearn.pipeline.util import get_dataset, \
    _test_classifier_predict_proba

from .test_base import BaseClassificationComponentTest


class LibSVM_SVCComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.96
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 0.32242983456012941
    res["default_iris_sparse"] = 0.64
    res["default_digits"] = 0.096539162112932606
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.90103217972070426
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.svm.SVC
    module = LibSVM_SVC

    def test_default_configuration_predict_proba_individual(self):
        # Leave this additional test here
        for i in range(2):
            predictions, targets = _test_classifier_predict_proba(
                LibSVM_SVC, sparse=True, dataset='digits',
                train_size_maximum=500)
            self.assertAlmostEqual(5.4706296711768925,
                                   sklearn.metrics.log_loss(targets,
                                                            predictions))

        for i in range(2):
            predictions, targets = _test_classifier_predict_proba(
                LibSVM_SVC, sparse=True, dataset='iris')
            self.assertAlmostEqual(0.84336416900751887,
                                   sklearn.metrics.log_loss(targets,
                                                            predictions))

        # 2 class
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
            remove_training_data = Y_train == 2
            remove_test_data = Y_test == 2
            X_train = X_train[~remove_training_data]
            Y_train = Y_train[~remove_training_data]
            X_test = X_test[~remove_test_data]
            Y_test = Y_test[~remove_test_data]
            ss = sklearn.preprocessing.StandardScaler()
            X_train = ss.fit_transform(X_train)
            configuration_space = LibSVM_SVC.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            cls = LibSVM_SVC(random_state=1, **{hp_name: default[hp_name]
                                                for hp_name in default
                                                if default[hp_name] is not None})

            cls = cls.fit(X_train, Y_train)
            prediction = cls.predict_proba(X_test)
            self.assertAlmostEqual(sklearn.metrics.log_loss(Y_test, prediction),
                                   0.6932, places=4)