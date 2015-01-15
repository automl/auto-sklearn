__author__ = 'feurerm'

import copy
import numpy as np
import StringIO
import unittest

import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.svm

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from AutoSklearn.classification import AutoSklearnClassifier
from AutoSklearn.components.classification_base import AutoSklearnClassificationAlgorithm
from AutoSklearn.components.preprocessor_base import AutoSklearnPreprocessingAlgorithm
import AutoSklearn.components.classification as classification_components
import AutoSklearn.components.preprocessing as preprocessing_components
from AutoSklearn.util import get_dataset

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

    def test_default_configuration(self):
        for i in range(2):
            cs = AutoSklearnClassifier.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
            auto = AutoSklearnClassifier(default)
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(X_test)
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, Y_test))
            scores = auto.predict_proba(X_test)

    def test_get_hyperparameter_search_space(self):
        cs = AutoSklearnClassifier.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(67, len(hyperparameters))
        self.assertEqual(len(hyperparameters) - 4, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = AutoSklearnClassifier.get_hyperparameter_search_space(
            include_estimators=['libsvm_svc'])
        self.assertEqual(cs.get_hyperparameter('classifier'),
            CategoricalHyperparameter('classifier', ['libsvm_svc']))

        cs = AutoSklearnClassifier.get_hyperparameter_search_space(
            exclude_estimators=['libsvm_svc'])
        self.assertNotIn('libsvm_svc', str(cs))

        cs = AutoSklearnClassifier.get_hyperparameter_search_space(
            include_preprocessors=['pca'])
        self.assertEqual(cs.get_hyperparameter('preprocessor'),
            CategoricalHyperparameter('preprocessor', ["None", 'pca']))

        cs = AutoSklearnClassifier.get_hyperparameter_search_space(
            exclude_preprocessors=['pca'])
        self.assertNotIn('pca', str(cs))

    def test_get_hyperparameter_search_space_dataset_properties(self):
        full_cs = AutoSklearnClassifier.get_hyperparameter_search_space()
        cs_mc = AutoSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'multiclass': True})
        self.assertEqual(full_cs, cs_mc)

        cs_ml = AutoSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'multilabel': True})
        self.assertNotIn('k_nearest_neighbors', str(cs_ml))
        self.assertNotIn('liblinear', str(cs_ml))
        self.assertNotIn('libsvm_svc', str(cs_ml))
        self.assertNotIn('sgd', str(cs_ml))

        cs_sp = AutoSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        self.assertNotIn('extra_trees', str(cs_sp))
        self.assertNotIn('gradient_boosting', str(cs_sp))
        self.assertNotIn('random_forest', str(cs_sp))

        cs_mc_ml = AutoSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'multilabel': True, 'multiclass': True})
        self.assertEqual(cs_ml, cs_mc_ml)

        self.assertRaisesRegexp(ValueError,
                                "No classifier to build a configuration space "
                                "for...", AutoSklearnClassifier.
                                get_hyperparameter_search_space,
                                dataset_properties={'multilabel': True,
                                                    'multiclass': True,
                                                    'sparse': True})

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
