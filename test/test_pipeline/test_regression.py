__author__ = 'eggenspk'

import copy
import resource
import traceback
import unittest

import mock
import numpy as np
import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.svm
from sklearn.utils.testing import assert_array_almost_equal

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.regression import SimpleRegressionPipeline
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm, AutoSklearnRegressionAlgorithm
import autosklearn.pipeline.components.regression as regression_components
import autosklearn.pipeline.components.feature_preprocessing as preprocessing_components
from autosklearn.pipeline.util import get_dataset
from autosklearn.pipeline.constants import *


class SimpleRegressionPipelineTest(unittest.TestCase):

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
            self.assertIsInstance(output, tuple)
            for i in inp:
                self.assertIn(i, (SPARSE, DENSE, SIGNED_DATA, UNSIGNED_DATA))
            self.assertEqual(output, (PREDICTIONS,))
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
            self.assertIn(AutoSklearnRegressionAlgorithm,
                            regressors[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            if hasattr(preprocessors[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnPreprocessingAlgorithm,
                            preprocessors[key].__bases__)

    def test_configurations(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        cs = SimpleRegressionPipeline.get_hyperparameter_search_space()

        print(cs)
        cs.seed(1)

        for i in range(10):
            config = cs.sample_configuration()
            config._populate_values()
            if config['regressor:sgd:n_iter'] is not None:
                config._values['regressor:sgd:n_iter'] = 5

            X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston')
            cls = SimpleRegressionPipeline(config, random_state=1)
            print(config)
            try:
                cls.fit(X_train, Y_train)
                X_test_ = X_test.copy()
                predictions = cls.predict(X_test)
                self.assertIsInstance(predictions, np.ndarray)
                predicted_probabiliets = cls.predict(X_test_)
                self.assertIsInstance(predicted_probabiliets, np.ndarray)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                        e.args[0]:
                    continue
                elif "all features are discarded" in e.args[0]:
                    continue
                elif "removed all features" in e.args[0]:
                    continue
                elif "Bug in scikit-learn:" in e.args[0]:
                    continue
                else:
                    print(config)
                    print(traceback.format_exc())
                    raise e
            except RuntimeWarning as e:
                if "invalid value encountered in sqrt" in e.args[0]:
                    continue
                elif "divide by zero encountered in" in e.args[0]:
                    continue
                elif "invalid value encountered in divide" in e.args[0]:
                    continue
                elif "invalid value encountered in true_divide" in e.args[0]:
                    continue
                else:
                    print(config)
                    print(traceback.format_exc())
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.args[0]:
                    continue
                else:
                    print(config)
                    print(traceback.format_exc())
                    raise e
            except MemoryError as e:
                continue

    def test_default_configuration(self):
        for i in range(2):
            cs = SimpleRegressionPipeline.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes')
            auto = SimpleRegressionPipeline(default)
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(copy.deepcopy(X_test))
            # The lower the worse
            r2_score = sklearn.metrics.r2_score(Y_test, predictions)
            self.assertAlmostEqual(0.41626416529791199, r2_score)
            model_score = auto.score(copy.deepcopy(X_test), Y_test)
            self.assertEqual(model_score, r2_score)

    def test_repr(self):
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        representation = repr(SimpleRegressionPipeline(default))
        cls = eval(representation)
        self.assertIsInstance(cls, SimpleRegressionPipeline)

    def test_get_hyperparameter_search_space(self):
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(130, len(hyperparameters))
        self.assertEqual(len(hyperparameters) - 5, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            include={'regressor': ['random_forest']})
        self.assertEqual(cs.get_hyperparameter('regressor:__choice__'),
            CategoricalHyperparameter('regressor:__choice__', ['random_forest']))

        # TODO add this test when more than one regressor is present
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            exclude={'regressor': ['random_forest']})
        self.assertNotIn('random_forest', str(cs))

        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            include={'preprocessor': ['pca']})
        self.assertEqual(cs.get_hyperparameter('preprocessor:__choice__'),
            CategoricalHyperparameter('preprocessor:__choice__', ['pca']))

        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            exclude={'preprocessor': ['no_preprocessing']})
        self.assertNotIn('no_preprocessing', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(
            self):
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            include={'preprocessor': ['densifier']},
            dataset_properties={'sparse': True})
        self.assertEqual(cs.get_hyperparameter('regressor:__choice__').default,
                         'gradient_boosting')

        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            include={'preprocessor': ['nystroem_sampler']})
        self.assertEqual(cs.get_hyperparameter('regressor:__choice__').default,
                         'sgd')

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(ValueError, "Cannot find a legal default "
                                            "configuration.",
                                SimpleRegressionPipeline.get_hyperparameter_search_space,
                                include={'regressor': ['random_forest'],
                                         'preprocessor': ['kitchen_sinks']})

        # It must also be catched that no classifiers which can handle sparse
        # data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Cannot find a legal default "
                                            "configuration",
                                SimpleRegressionPipeline.get_hyperparameter_search_space,
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
        full_cs = SimpleRegressionPipeline.get_hyperparameter_search_space()
        cs_mc = SimpleRegressionPipeline.get_hyperparameter_search_space()
        self.assertEqual(full_cs, cs_mc)

        cs_ml = SimpleRegressionPipeline.get_hyperparameter_search_space()
        self.assertNotIn('k_nearest_neighbors', str(cs_ml))
        self.assertNotIn('liblinear', str(cs_ml))
        self.assertNotIn('libsvm_svc', str(cs_ml))
        self.assertNotIn('sgd', str(cs_ml))

        cs_sp = SimpleRegressionPipeline.get_hyperparameter_search_space(
            sparse=True)
        self.assertNotIn('extra_trees', str(cs_sp))
        self.assertNotIn('gradient_boosting', str(cs_sp))
        self.assertNotIn('random_forest', str(cs_sp))

        cs_mc_ml = SimpleRegressionPipeline.get_hyperparameter_search_space()
        self.assertEqual(cs_ml, cs_mc_ml)

        self.assertRaisesRegexp(ValueError,
                                "No regressor to build a configuration space "
                                "for...", SimpleRegressionPipeline.
                                get_hyperparameter_search_space,
                                multiclass=True, multilabel=True, sparse=True)
    """

    def test_predict_batched(self):
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        cls = SimpleRegressionPipeline(default)

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
        cs = SimpleRegressionPipeline.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        default = cs.get_default_configuration()
        cls = SimpleRegressionPipeline(default)

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
