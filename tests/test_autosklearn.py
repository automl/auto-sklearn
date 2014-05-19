__author__ = 'feurerm'

import numpy as np
import StringIO
import unittest

import hyperopt

import sklearn.datasets
import sklearn.decomposition

from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.components.classification_base import AutoSklearnClassificationAlgorithm
from AutoSklearn.components.preprocessor_base import AutoSklearnPreprocessingAlgorithm
import AutoSklearn.components.classification as classification_components
import AutoSklearn.components.preprocessing as preprocessing_components
from AutoSklearn.util import NoModelException

class TestAutoSKlearnClassifier(unittest.TestCase):
    def get_iris(self):
        iris = sklearn.datasets.load_iris()
        X = iris.data
        Y = iris.target
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        rs.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        X_train = X[:100]
        Y_train = Y[:100]
        X_test = X[100:]
        Y_test = Y[100:]
        return X_train, Y_train, X_test, Y_test

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

    def test_init_no_classifier(self):
        try:
            AutoSklearnClassifier(None, None)
        except NoModelException as e:
            self.assertEqual(e.__str__(),
            '"You called <class \'AutoSklearn.autosklearn'
            '.AutoSklearnClassifier\'>.__init__() without '
            'specifying a model first."')

    def test_init_unknown_classifier(self):
        self.assertRaises(KeyError, AutoSklearnClassifier,
                          "qufrpdvltromeaiudtroembdtaiubo", None)

    @unittest.skip("test_init_parameters_as_dict_or_as_keywords Not yet Implemented")
    def test_init_parameters_as_dict_or_as_keywords(self):
        pass

    def test_fit_iris(self):
        auto = AutoSklearnClassifier("liblinear", None)
        X_train, Y_train, X_test, Y_test = self.get_iris()
        auto = auto.fit(X_train, Y_train)
        self.assertIsInstance(auto, AutoSklearnClassifier)
        self.assertIsInstance(auto._estimator, AutoSklearnClassificationAlgorithm)

    def test_predict_iris(self):
        auto = AutoSklearnClassifier("liblinear", None)
        X_train, Y_train, X_test, Y_test = self.get_iris()
        auto = auto.fit(X_train, Y_train)
        predictions = auto.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_fit_with_preproc(self):
        auto = AutoSklearnClassifier("liblinear", "pca")
        X_train, Y_train, X_test, Y_test = self.get_iris()
        auto = auto.fit(X_train, Y_train)
        self.assertIsInstance(auto, AutoSklearnClassifier)
        self.assertIsInstance(auto._preprocessor, AutoSklearnPreprocessingAlgorithm)
        self.assertIsInstance(auto._preprocessor.preprocessor, sklearn
                              .decomposition.PCA)

    def test_predict_with_preproc(self):
        auto = AutoSklearnClassifier("liblinear", "pca")
        X_train, Y_train, X_test, Y_test = self.get_iris()
        auto = auto.fit(X_train, Y_train)
        prediction = auto.predict(X_test)
        self.assertIsInstance(auto, AutoSklearnClassifier)
        self.assertIsInstance(auto._preprocessor, AutoSklearnPreprocessingAlgorithm)

    def test_specify_hyperparameters(self):
        auto = AutoSklearnClassifier(random_state=1,
            parameters={"classifier": "random_forest", "preprocessing":
                "pca", "random_forest:n_estimators": 1,
                "random_forest:max_features": 1.0})
        X_train, Y_train, X_test, Y_test = self.get_iris()
        auto = auto.fit(X_train, Y_train)
        predictions = auto.predict(X_test)
        accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
        self.assertAlmostEqual(accuracy, 0.939999999)
        self.assertEqual(auto._estimator.estimator.n_estimators, 1)

    def test_get_hyperparameter_search_space(self):
        auto = AutoSklearnClassifier(None, None)
        space = auto.get_hyperparameter_search_space()
        space = hyperopt.pyll.base.as_apply(space)
        print space

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