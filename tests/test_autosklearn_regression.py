__author__ = 'eggenspk'

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

from AutoSklearn.autosklearn_regression import AutoSklearnRegressor
from AutoSklearn.components.regression_base import AutoSklearnRegressionAlgorithm
from AutoSklearn.components.preprocessor_base import AutoSklearnPreprocessingAlgorithm
import AutoSklearn.components.regression as regression_components
import AutoSklearn.components.preprocessing as preprocessing_components
from AutoSklearn.util import get_dataset


class TestAutoSKlearnRegressor(unittest.TestCase):
    # TODO: test for both possible ways to initialize AutoSklearn
    # parameters and other...

    def test_find_classifiers(self):
        regressors = regression_components._regressors
        self.assertGreaterEqual(len(regressors), 1)
        for key in regressors:
            self.assertIn(AutoSklearnRegressionAlgorithm,
                            regressors[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            self.assertIn(AutoSklearnPreprocessingAlgorithm,
                            preprocessors[key].__bases__)

    def test_default_configuration(self):
        for i in range(2):
            cs = AutoSklearnRegressor.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes')
            auto = AutoSklearnRegressor(default)
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(copy.deepcopy(X_test))
            # The lower the worse
            r2_score = sklearn.metrics.r2_score(Y_test, predictions)
            self.assertAlmostEqual(0.41855369945075482, r2_score)
            model_score = auto.score(copy.deepcopy(X_test), Y_test)
            self.assertEqual(model_score, r2_score)

    def test_get_hyperparameter_search_space(self):
        cs = AutoSklearnRegressor.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(18, len(hyperparameters))
        self.assertEqual(len(hyperparameters) - 4, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = AutoSklearnRegressor.get_hyperparameter_search_space(
            include_regressors=['random_forest'])
        self.assertEqual(cs.get_hyperparameter('regressor'),
            CategoricalHyperparameter('regressor', ['random_forest']))

        # TODO add this test when more than one regressor is present
        cs = AutoSklearnRegressor.get_hyperparameter_search_space(
            exclude_regressors=['random_forest'])
        self.assertNotIn('random_forest', str(cs))

        cs = AutoSklearnRegressor.get_hyperparameter_search_space(
            include_preprocessors=['pca'])
        self.assertEqual(cs.get_hyperparameter('preprocessor'),
            CategoricalHyperparameter('preprocessor', ["None", 'pca']))

        cs = AutoSklearnRegressor.get_hyperparameter_search_space(
            exclude_preprocessors=['pca'])
        self.assertNotIn('pca', str(cs))

    @unittest.skip("test_get_hyperparameter_search_space_dataset_properties" +
                   " Not yet Implemented")
    def test_get_hyperparameter_search_space_dataset_properties(self):
        # TODO: We do not have any dataset properties for regression, so this
        # test is somewhat stupid
        pass
        """
        full_cs = AutoSklearnRegressor.get_hyperparameter_search_space()
        cs_mc = AutoSklearnRegressor.get_hyperparameter_search_space()
        self.assertEqual(full_cs, cs_mc)

        cs_ml = AutoSklearnRegressor.get_hyperparameter_search_space()
        self.assertNotIn('k_nearest_neighbors', str(cs_ml))
        self.assertNotIn('liblinear', str(cs_ml))
        self.assertNotIn('libsvm_svc', str(cs_ml))
        self.assertNotIn('sgd', str(cs_ml))

        cs_sp = AutoSklearnRegressor.get_hyperparameter_search_space(
            sparse=True)
        self.assertNotIn('extra_trees', str(cs_sp))
        self.assertNotIn('gradient_boosting', str(cs_sp))
        self.assertNotIn('random_forest', str(cs_sp))

        cs_mc_ml = AutoSklearnRegressor.get_hyperparameter_search_space()
        self.assertEqual(cs_ml, cs_mc_ml)

        self.assertRaisesRegexp(ValueError,
                                "No regressor to build a configuration space "
                                "for...", AutoSklearnRegressor.
                                get_hyperparameter_search_space,
                                multiclass=True, multilabel=True, sparse=True)
    """

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