import unittest

from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
from autosklearn.pipeline.util import _test_classifier, \
    _test_classifier_predict_proba, get_dataset

import numpy as np
import sklearn.metrics
import sklearn.svm


class LibSVM_SVCComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        for i in range(2):
            predictions, targets = _test_classifier(LibSVM_SVC, dataset='iris')
            self.assertAlmostEqual(0.96,
                sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_predict_proba(self):
        for i in range(2):
            predictions, targets = _test_classifier_predict_proba(
                LibSVM_SVC, sparse=True, dataset='digits',
                train_size_maximum=500)
            self.assertAlmostEqual(4.6680593525563063,
                                   sklearn.metrics.log_loss(targets,
                                                            predictions))

        for i in range(2):
            predictions, targets = _test_classifier_predict_proba(
                LibSVM_SVC, sparse=True, dataset='iris')
            self.assertAlmostEqual(0.8649665185853217,
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
                                   0.69323680119641773)

    def test_default_configuration_binary(self):
        for i in range(2):
            predictions, targets = _test_classifier(LibSVM_SVC,
                                                    make_binary=True)
            self.assertAlmostEqual(1.0,
                                   sklearn.metrics.accuracy_score(
                                       predictions, targets))

    def test_target_algorithm_multioutput_multiclass_support(self):
        cls = sklearn.svm.SVC()

        X = np.random.random((10, 10))
        y = np.random.randint(0, 1, size=(10, 10))
        self.assertRaisesRegexp(ValueError, 'bad input shape \(10, 10\)',
                                cls.fit, X, y)
