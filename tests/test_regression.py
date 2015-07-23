__author__ = 'eggenspk'

import copy
import unittest

import mock
import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.svm
from sklearn.utils.testing import assert_array_almost_equal

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ParamSklearn.regression import ParamSklearnRegressor
from ParamSklearn.components.base import ParamSklearnRegressionAlgorithm
from ParamSklearn.components.base import ParamSklearnPreprocessingAlgorithm
import ParamSklearn.components.regression as regression_components
import ParamSklearn.components.preprocessing as preprocessing_components
from ParamSklearn.util import get_dataset, SPARSE, DENSE, PREDICTIONS


class TestParamSKlearnRegressor(unittest.TestCase):

    def test_io_dict(self):
        regressors = regression_components._regressors
        for r in regressors:
            if regressors[r] == regression_components.RegressorChoice:
                continue
            props = regressors[r].get_properties()
            self.assertIn('input', props)
            self.assertIn('output', props)
            inp = props['input']
            output = props['output']

            self.assertIsInstance(inp, tuple)
            self.assertIsInstance(output, str)
            for i in inp:
                self.assertIn(i, (SPARSE, DENSE))
            self.assertEqual(output, PREDICTIONS)
            self.assertIn('handles_regression', props)
            self.assertTrue(props['handles_regression'])
            self.assertIn('handles_classification', props)
            self.assertIn('handles_multiclass', props)
            self.assertIn('handles_multilabel', props)
            self.assertFalse(props['handles_classification'])
            self.assertFalse(props['handles_multiclass'])
            self.assertFalse(props['handles_multilabel'])

    def test_find_regressors(self):
        regressors = regression_components._regressors
        self.assertGreaterEqual(len(regressors), 1)
        for key in regressors:
            if hasattr(regressors[key], 'get_components'):
                continue
            self.assertIn(ParamSklearnRegressionAlgorithm,
                            regressors[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            if hasattr(preprocessors[key], 'get_components'):
                continue
            self.assertIn(ParamSklearnPreprocessingAlgorithm,
                            preprocessors[key].__bases__)

    def test_default_configuration(self):
        for i in range(2):
            cs = ParamSklearnRegressor.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes')
            auto = ParamSklearnRegressor(default)
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(copy.deepcopy(X_test))
            # The lower the worse
            r2_score = sklearn.metrics.r2_score(Y_test, predictions)
            self.assertAlmostEqual(0.41211271098191482, r2_score)
            model_score = auto.score(copy.deepcopy(X_test), Y_test)
            self.assertEqual(model_score, r2_score)

    def test_get_hyperparameter_search_space(self):
        cs = ParamSklearnRegressor.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(82, len(hyperparameters))
        self.assertEqual(len(hyperparameters) - 4, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = ParamSklearnRegressor.get_hyperparameter_search_space(
            include={'regressor': ['random_forest']})
        self.assertEqual(cs.get_hyperparameter('regressor:__choice__'),
            CategoricalHyperparameter('regressor:__choice__', ['random_forest']))

        # TODO add this test when more than one regressor is present
        cs = ParamSklearnRegressor.get_hyperparameter_search_space(
            exclude={'regressor': ['random_forest']})
        self.assertNotIn('random_forest', str(cs))

        cs = ParamSklearnRegressor.get_hyperparameter_search_space(
            include={'preprocessor': ['pca']})
        self.assertEqual(cs.get_hyperparameter('preprocessor:__choice__'),
            CategoricalHyperparameter('preprocessor:__choice__', ['pca']))

        cs = ParamSklearnRegressor.get_hyperparameter_search_space(
            exclude={'preprocessor': ['no_preprocessing']})
        self.assertNotIn('no_preprocessing', str(cs))

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  imputation:strategy, Value: mean\n"
            "  preprocessor:__choice__, Value: kitchen_sinks\n"
            "  preprocessor:kitchen_sinks:gamma, Value: 1.0\n"
            "  preprocessor:kitchen_sinks:n_components, Value: 100\n"
            "  regressor:__choice__, Value: random_forest\n"
            "  regressor:random_forest:bootstrap, Value: True\n"
            "  regressor:random_forest:criterion, Constant: mse\n"
            "  regressor:random_forest:max_depth, Constant: None\n"
            "  regressor:random_forest:max_features, Value: 1.0\n"
            "  regressor:random_forest:min_samples_leaf, Value: 1\n"
            "  regressor:random_forest:min_samples_split, Value: 2\n"
            "  regressor:random_forest:n_estimators, Constant: 100\n"
            "  rescaling:strategy, Value: min/max\n"
            "violates forbidden clause \(Forbidden: regressor:__choice__ == random_forest"
            " && Forbidden: preprocessor:__choice__ == kitchen_sinks\)",
                                ParamSklearnRegressor.get_hyperparameter_search_space,
                                include={'regressor': ['random_forest'],
                                         'preprocessor': ['kitchen_sinks']})

        # It must also be catched that no classifiers which can handle sparse
        # data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  imputation:strategy, Value: mean\n"
            "  preprocessor:__choice__, Value: densifier\n"
            "  regressor:__choice__, Value: ridge_regression\n"
            "  regressor:ridge_regression:alpha, Value: 1.0\n"
            "  rescaling:strategy, Value: min/max\n"
            "violates forbidden clause \(Forbidden: regressor:__choice__ == "
            "ridge_regression && Forbidden: preprocessor:__choice__ == densifier\)",
                                ParamSklearnRegressor.get_hyperparameter_search_space,
                                include={'regressor': ['ridge_regression'],
                                         'preprocessor': ['densifier']},
                                dataset_properties={'sparse': True})

    @unittest.skip("test_get_hyperparameter_search_space_dataset_properties" +
                   " Not yet Implemented")
    def test_get_hyperparameter_search_space_dataset_properties(self):
        # TODO: We do not have any dataset properties for regression, so this
        # test is somewhat stupid
        pass
        """
        full_cs = ParamSklearnRegressor.get_hyperparameter_search_space()
        cs_mc = ParamSklearnRegressor.get_hyperparameter_search_space()
        self.assertEqual(full_cs, cs_mc)

        cs_ml = ParamSklearnRegressor.get_hyperparameter_search_space()
        self.assertNotIn('k_nearest_neighbors', str(cs_ml))
        self.assertNotIn('liblinear', str(cs_ml))
        self.assertNotIn('libsvm_svc', str(cs_ml))
        self.assertNotIn('sgd', str(cs_ml))

        cs_sp = ParamSklearnRegressor.get_hyperparameter_search_space(
            sparse=True)
        self.assertNotIn('extra_trees', str(cs_sp))
        self.assertNotIn('gradient_boosting', str(cs_sp))
        self.assertNotIn('random_forest', str(cs_sp))

        cs_mc_ml = ParamSklearnRegressor.get_hyperparameter_search_space()
        self.assertEqual(cs_ml, cs_mc_ml)

        self.assertRaisesRegexp(ValueError,
                                "No regressor to build a configuration space "
                                "for...", ParamSklearnRegressor.
                                get_hyperparameter_search_space,
                                multiclass=True, multilabel=True, sparse=True)
    """

    def test_predict_batched(self):
        cs = ParamSklearnRegressor.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        cls = ParamSklearnRegressor(default)

        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston')
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_)
        cls.pipeline_ = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((356,), prediction.shape)
        self.assertEqual(18, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_batched_sparse(self):
        cs = ParamSklearnRegressor.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        default = cs.get_default_configuration()
        cls = ParamSklearnRegressor(default)

        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_)
        cls.pipeline_ = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((356,), prediction.shape)
        self.assertEqual(18, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

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
