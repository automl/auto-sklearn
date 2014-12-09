__author__ = 'feurerm'

import numpy as np
import StringIO
import unittest

import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.svm

from HPOlibConfigSpace.configuration_space import Configuration, ConfigurationSpace

from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.components.classification_base import AutoSklearnClassificationAlgorithm
from AutoSklearn.components.preprocessor_base import AutoSklearnPreprocessingAlgorithm
import AutoSklearn.components.classification as classification_components
import AutoSklearn.components.preprocessing as preprocessing_components
from AutoSklearn.util import get_iris

class TestAutoSKlearnClassifier(unittest.TestCase):
    # TODO: test for both possible ways to initialize AutoSklearn
    # parameters and other...


    def test_find_classifiers(self):
        classifiers = classification_components._classifiers
        self.assertGreaterEqual(len(classifiers), 1)
        for key in classifiers:
            self.assertIn(AutoSklearnClassificationAlgorithm,
                            classifiers[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            self.assertIn(AutoSklearnPreprocessingAlgorithm,
                            preprocessors[key].__bases__)

    def test_predict_iris(self):
        cs = AutoSklearnClassifier.get_hyperparameter_search_space()
        hyperparameters = {}
        hyperparameters['classifier'] = cs.get_hyperparameter(
            "classifier").instantiate("liblinear")
        hyperparameters['liblinear:C'] = cs.get_hyperparameter("liblinear:C").\
            instantiate(1.0)
        hyperparameters['liblinear:dual'] = cs.get_hyperparameter(
            'liblinear:dual').instantiate('False')
        hyperparameters['liblinear:loss'] = cs.get_hyperparameter(
            'liblinear:loss').instantiate('l2')
        hyperparameters['liblinear:penalty'] = cs.get_hyperparameter(
            'liblinear:penalty').instantiate('l2')
        hyperparameters['preprocessor'] = cs.get_hyperparameter(
            'preprocessor').instantiate('None')
        config = Configuration(cs, hyperparameters=hyperparameters)

        auto = AutoSklearnClassifier(config)
        X_train, Y_train, X_test, Y_test = get_iris()
        auto = auto.fit(X_train, Y_train)
        predictions = auto.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
        self.assertIsInstance(auto, AutoSklearnClassifier)
        self.assertIsInstance(auto._estimator, AutoSklearnClassificationAlgorithm)
        self.assertIsInstance(auto._estimator.estimator, sklearn.svm.LinearSVC)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_fit_with_preproc(self):
        cs = AutoSklearnClassifier.get_hyperparameter_search_space()
        hyperparameters = {}
        hyperparameters['classifier'] = cs.get_hyperparameter(
            "classifier").instantiate("liblinear")
        hyperparameters['liblinear:C'] = cs.get_hyperparameter("liblinear:C"). \
            instantiate(1.0)
        hyperparameters['liblinear:dual'] = cs.get_hyperparameter(
            'liblinear:dual').instantiate('False')
        hyperparameters['liblinear:loss'] = cs.get_hyperparameter(
            'liblinear:loss').instantiate('l2')
        hyperparameters['liblinear:penalty'] = cs.get_hyperparameter(
            'liblinear:penalty').instantiate('l2')
        hyperparameters['preprocessor'] = cs.get_hyperparameter(
            'preprocessor').instantiate('pca')
        hyperparameters['pca:keep_variance'] = cs.get_hyperparameter(
            'pca:keep_variance').instantiate(1.0)
        hyperparameters['pca:whiten'] = cs.get_hyperparameter(
            'pca:whiten').instantiate('False')
        config = Configuration(cs, hyperparameters=hyperparameters)

        auto = AutoSklearnClassifier(config)
        X_train, Y_train, X_test, Y_test = get_iris()
        auto = auto.fit(X_train, Y_train)
        self.assertIsInstance(auto, AutoSklearnClassifier)
        self.assertIsInstance(auto._preprocessor, AutoSklearnPreprocessingAlgorithm)
        self.assertIsInstance(auto._preprocessor.preprocessor, sklearn
                              .decomposition.PCA)

        prediction = auto.predict(X_test)

    def test_get_hyperparameter_search_space(self):
        config = AutoSklearnClassifier.get_hyperparameter_search_space()
        self.assertIsInstance(config, ConfigurationSpace)

    @unittest.skip("test_check_random_state Not yet Implemented")
    def test_check_random_state(self):
        raise NotImplementedError()

    @unittest.skip("test_validate_input_X Not yet Implemented")
    def test_validate_input_X(self):
        raise NotImplementedError()

    @unittest.skip("test_validate_input_Y Not yet Implemented")
    def test_validate_input_Y(self):
        raise NotImplementedError()

    def test_set_params(self):
        pass

    def test_get_params(self):
        pass