import copy
import resource
import sys
import traceback
import unittest
import unittest.mock

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
    _multiprocess_can_split_ = True

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
        cs = SimpleRegressionPipeline().get_hyperparameter_search_space()

        self._test_configurations(cs)

    def test_configurations_signed_data(self):
        dataset_properties = {'signed': True}
        cs = SimpleRegressionPipeline(dataset_properties=dataset_properties).\
            get_hyperparameter_search_space()

        self._test_configurations(configurations_space=cs,
                                  dataset_properties=dataset_properties)

    def test_configurations_sparse(self):
        dataset_properties = {'sparse': True}
        cs = SimpleRegressionPipeline(
            dataset_properties=dataset_properties
        ).get_hyperparameter_search_space()

        self._test_configurations(cs, make_sparse=True,
                                  dataset_properties=dataset_properties)

    def _test_configurations(self, configurations_space, make_sparse=False,
                             data=None, dataset_properties=None):
        # Use a limit of ~4GiB
        limit = 3072 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        configurations_space.seed(1)

        for i in range(10):
            config = configurations_space.sample_configuration()
            config._populate_values()

            # Restrict configurations which could take too long on travis-ci
            restrictions = {'regressor:adaboost:n_estimators': 50,
                            'regressor:adaboost:max_depth': 1,
                            'preprocessor:kernel_pca:n_components': 10,
                            'preprocessor:kitchen_sinks:n_components': 50,
                            'regressor:libsvm_svc:degree': 2,
                            'regressor:libsvm_svr:degree': 2,
                            'preprocessor:truncatedSVD:target_dim': 10,
                            'preprocessor:polynomial:degree': 2,
                            'regressor:lda:n_components': 10}

            for restrict_parameter in restrictions:
                restrict_to = restrictions[restrict_parameter]
                if restrict_parameter in config and \
                                config[restrict_parameter] is not None:
                    config._values[restrict_parameter] = restrict_to

            if data is None:
                X_train, Y_train, X_test, Y_test = get_dataset(
                    dataset='boston', make_sparse=make_sparse, add_NaNs=True)
            else:
                X_train = data['X_train'].copy()
                Y_train = data['Y_train'].copy()
                X_test = data['X_test'].copy()
                Y_test = data['Y_test'].copy()

            cls = SimpleRegressionPipeline(random_state=1,
                                           dataset_properties=dataset_properties)
            cls.set_hyperparameters(config)
            try:
                cls.fit(X_train, Y_train)
                predictions = cls.predict(X_test)
            except MemoryError as e:
                continue
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                        e.args[0]:
                    continue
                elif "removed all features" in e.args[0]:
                    continue
                elif "all features are discarded" in e.args[0]:
                    continue
                elif "Numerical problems in QDA" in e.args[0]:
                    continue
                elif 'Bug in scikit-learn' in e.args[0]:
                    continue
                elif 'The condensed distance matrix must contain only finite ' \
                     'values.' in e.args[0]:
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
                    traceback.print_tb(sys.exc_info()[2])
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.args[0]:
                    continue
                else:
                    print(config)
                    traceback.print_tb(sys.exc_info()[2])
                    raise e
            except Exception as e:
                if "Multiple input features cannot have the same target value" in e.args[0]:
                    continue
                else:
                    print(config)
                    traceback.print_tb(sys.exc_info()[2])
                    raise e

    def test_default_configuration(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='diabetes')
            auto = SimpleRegressionPipeline()
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(copy.deepcopy(X_test))
            # The lower the worse
            r2_score = sklearn.metrics.r2_score(Y_test, predictions)
            self.assertAlmostEqual(0.339, r2_score, places=3)
            model_score = auto.score(copy.deepcopy(X_test), Y_test)
            self.assertAlmostEqual(model_score, r2_score, places=5)

    def test_default_configuration_iterative_fit(self):
        regressor = SimpleRegressionPipeline(
            include={'regressor': ['random_forest'],
                     'preprocessor': ['no_preprocessing']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston')
        XT = regressor.fit_transformer(X_train, Y_train)
        for i in range(1, 11):
            regressor.iterative_fit(X_train, Y_train)
            self.assertEqual(regressor.steps[-1][-1].choice.estimator.n_estimators,
                             i)

    def test_repr(self):
        representation = repr(SimpleRegressionPipeline())
        cls = eval(representation)
        self.assertIsInstance(cls, SimpleRegressionPipeline)

    def test_get_hyperparameter_search_space(self):
        cs = SimpleRegressionPipeline().get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(158, len(hyperparameters))
        self.assertEqual(len(hyperparameters) - 5, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = SimpleRegressionPipeline(
            include={'regressor': ['random_forest']}).get_hyperparameter_search_space()
        self.assertEqual(cs.get_hyperparameter('regressor:__choice__'),
            CategoricalHyperparameter('regressor:__choice__', ['random_forest']))

        # TODO add this test when more than one regressor is present
        cs = SimpleRegressionPipeline(exclude={'regressor': ['random_forest']}).\
            get_hyperparameter_search_space()
        self.assertNotIn('random_forest', str(cs))

        cs = SimpleRegressionPipeline(include={'preprocessor': ['pca']}).\
            get_hyperparameter_search_space()
        self.assertEqual(cs.get_hyperparameter('preprocessor:__choice__'),
            CategoricalHyperparameter('preprocessor:__choice__', ['pca']))

        cs = SimpleRegressionPipeline(exclude={'preprocessor': ['no_preprocessing']}).\
            get_hyperparameter_search_space()
        self.assertNotIn('no_preprocessing', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(
            self):
        cs = SimpleRegressionPipeline(include={'preprocessor': ['densifier']},
                                      dataset_properties={'sparse': True}).\
            get_hyperparameter_search_space()
        self.assertEqual(
            cs.get_hyperparameter('regressor:__choice__').default_value,
            'gradient_boosting'
        )

        cs = SimpleRegressionPipeline(include={'preprocessor': ['nystroem_sampler']}).\
            get_hyperparameter_search_space()
        self.assertEqual(
            cs.get_hyperparameter('regressor:__choice__').default_value,
            'sgd'
        )

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(ValueError, "Cannot find a legal default "
                                            "configuration.",
                                SimpleRegressionPipeline,
                                include={'regressor': ['random_forest'],
                                         'preprocessor': ['kitchen_sinks']})

        # It must also be catched that no classifiers which can handle sparse
        # data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Cannot find a legal default "
                                            "configuration",
                                SimpleRegressionPipeline,
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
        include = {'regressor': ['decision_tree']}
        cs = SimpleRegressionPipeline(include=include).get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        regressor = SimpleRegressionPipeline(default, include=include)

        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston')
        regressor.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = regressor.predict(X_test_)
        mock_predict = unittest.mock.Mock(wraps=regressor.steps[-1][-1].predict)
        regressor.steps[-1][-1].predict = mock_predict
        prediction = regressor.predict(X_test, batch_size=20)
        self.assertEqual((356,), prediction.shape)
        self.assertEqual(18, mock_predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_batched_sparse(self):
        dataset_properties = {'sparse': True}
        include = {'regressor': ['decision_tree']}
        cs = SimpleRegressionPipeline(dataset_properties=dataset_properties,
                                      include=include).get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        regressor = SimpleRegressionPipeline(default,
                                             dataset_properties=dataset_properties,
                                             include=include)

        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston',
                                                       make_sparse=True)
        regressor.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = regressor.predict(X_test_)
        mock_predict = unittest.mock.Mock(wraps=regressor.steps[-1][-1].predict)
        regressor.steps[-1][-1].predict = mock_predict
        prediction = regressor.predict(X_test, batch_size=20)
        self.assertEqual((356,), prediction.shape)
        self.assertEqual(18, mock_predict.call_count)
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
