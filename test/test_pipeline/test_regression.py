import copy
import itertools
import resource
import sys
import tempfile
import traceback
import unittest
import unittest.mock

from joblib import Memory
import numpy as np
import sklearn.datasets
import sklearn.decomposition
from sklearn.base import clone
import sklearn.ensemble
import sklearn.svm
from sklearn.utils.validation import check_is_fitted

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.regression import SimpleRegressionPipeline
from autosklearn.pipeline.components.base import \
    AutoSklearnPreprocessingAlgorithm, AutoSklearnRegressionAlgorithm
import autosklearn.pipeline.components.regression as regression_components
from autosklearn.pipeline.components.base import AutoSklearnComponent, AutoSklearnChoice
import autosklearn.pipeline.components.feature_preprocessing as preprocessing_components
from autosklearn.pipeline.util import get_dataset
from autosklearn.pipeline.constants import SPARSE, DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS


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
            self.assertIn('handles_multioutput', props)
            self.assertFalse(props['handles_classification'])
            self.assertFalse(props['handles_multiclass'])
            self.assertFalse(props['handles_multilabel'])

    def test_find_regressors(self):
        regressors = regression_components._regressors
        self.assertGreaterEqual(len(regressors), 1)
        for key in regressors:
            if hasattr(regressors[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnRegressionAlgorithm, regressors[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            if hasattr(preprocessors[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnPreprocessingAlgorithm, preprocessors[key].__bases__)

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

    def test_multioutput(self):
        cache = Memory(location=tempfile.gettempdir())
        cached_func = cache.cache(
            sklearn.datasets.make_regression
        )
        X, Y = cached_func(
            n_samples=250,
            n_features=20,
            n_informative=9,
            n_targets=4,
            bias=0.5,
            effective_rank=10,
            tail_strength=0.4,
            noise=0.3,
            shuffle=True,
            coef=False,
            random_state=1
        )
        X_train = X[:200, :]
        Y_train = Y[:200, :]
        X_test = X[200:, :]
        Y_test = Y[200:, :]

        data = {'X_train': X_train, 'Y_train': Y_train,
                'X_test': X_test, 'Y_test': Y_test}

        dataset_properties = {'multioutput': True}
        cs = SimpleRegressionPipeline(dataset_properties=dataset_properties).\
            get_hyperparameter_search_space()
        self._test_configurations(cs, data=data,
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
                            'feature_preprocessor:kernel_pca:n_components': 10,
                            'feature_preprocessor:kitchen_sinks:n_components': 50,
                            'regressor:libsvm_svc:degree': 2,
                            'regressor:libsvm_svr:degree': 2,
                            'regressor:libsvm_svr:C': 1.,
                            'feature_preprocessor:truncatedSVD:target_dim': 10,
                            'feature_preprocessor:polynomial:degree': 2,
                            'regressor:lda:n_components': 10}

            for restrict_parameter in restrictions:
                restrict_to = restrictions[restrict_parameter]
                if restrict_parameter in config and config[restrict_parameter] is not None:
                    config._values[restrict_parameter] = restrict_to

            if data is None:
                X_train, Y_train, X_test, Y_test = get_dataset(
                    dataset='boston', make_sparse=make_sparse, add_NaNs=True)
            else:
                X_train = data['X_train'].copy()
                Y_train = data['Y_train'].copy()
                X_test = data['X_test'].copy()
                data['Y_test'].copy()

            cls = SimpleRegressionPipeline(random_state=1,
                                           dataset_properties=dataset_properties)
            cls.set_hyperparameters(config)

            # First make sure that for this configuration, setting the parameters
            # does not mistakenly set the estimator as fitted
            for name, step in cls.named_steps.items():
                with self.assertRaisesRegex(sklearn.exceptions.NotFittedError,
                                            "instance is not fitted yet"):
                    check_is_fitted(step)

            try:
                cls.fit(X_train, Y_train)
                # After fit, all components should be tagged as fitted
                # by sklearn. Check is fitted raises an exception if that
                # is not the case
                try:
                    for name, step in cls.named_steps.items():
                        check_is_fitted(step)
                except sklearn.exceptions.NotFittedError:
                    self.fail("config={} raised NotFittedError unexpectedly!".format(
                        config
                    ))

                cls.predict(X_test)
            except MemoryError:
                continue
            except np.linalg.LinAlgError:
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
                elif "invalid value encountered in multiply" in e.args[0]:
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
            self.assertAlmostEqual(0.3458397471855429, r2_score, places=2)
            model_score = auto.score(copy.deepcopy(X_test), Y_test)
            self.assertAlmostEqual(model_score, r2_score, places=5)

    def test_default_configuration_iterative_fit(self):
        regressor = SimpleRegressionPipeline(
            include={'regressor': ['random_forest'],
                     'feature_preprocessor': ['no_preprocessing']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston')
        regressor.fit_transformer(X_train, Y_train)
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
        forbiddens = cs.get_forbiddens()
        self.assertEqual(155, len(hyperparameters))
        self.assertEqual(len(hyperparameters) - 6, len(conditions))
        self.assertEqual(len(forbiddens), 35)

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = SimpleRegressionPipeline(
            include={'regressor': ['random_forest']}).get_hyperparameter_search_space()
        self.assertEqual(
            cs.get_hyperparameter('regressor:__choice__'),
            CategoricalHyperparameter('regressor:__choice__', ['random_forest']),
            )

        # TODO add this test when more than one regressor is present
        cs = SimpleRegressionPipeline(exclude={'regressor': ['random_forest']}).\
            get_hyperparameter_search_space()
        self.assertNotIn('random_forest', str(cs))

        cs = SimpleRegressionPipeline(include={'feature_preprocessor': ['pca']}).\
            get_hyperparameter_search_space()
        self.assertEqual(cs.get_hyperparameter(
            'feature_preprocessor:__choice__'),
            CategoricalHyperparameter('feature_preprocessor:__choice__', ['pca']))

        cs = SimpleRegressionPipeline(
            exclude={'feature_preprocessor': ['no_preprocessing']}).\
            get_hyperparameter_search_space()
        self.assertNotIn('no_preprocessing', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(
            self):
        cs = SimpleRegressionPipeline(include={'feature_preprocessor': ['densifier']},
                                      dataset_properties={'sparse': True}).\
            get_hyperparameter_search_space()
        self.assertEqual(
            cs.get_hyperparameter('regressor:__choice__').default_value,
            'gradient_boosting'
        )

        cs = SimpleRegressionPipeline(
            include={'feature_preprocessor': ['nystroem_sampler']}).\
            get_hyperparameter_search_space()
        self.assertEqual(
            cs.get_hyperparameter('regressor:__choice__').default_value,
            'sgd'
        )

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegex(
            ValueError,
            "Cannot find a legal default configuration.",
            SimpleRegressionPipeline,
            include={
                'regressor': ['random_forest'],
                'feature_preprocessor': ['kitchen_sinks']
            }
        )

        # It must also be catched that no classifiers which can handle sparse
        # data are located behind the densifier
        self.assertRaisesRegex(
            ValueError,
            "Cannot find a legal default configuration",
            SimpleRegressionPipeline,
            include={
                'regressor': ['extra_trees'],
                'feature_preprocessor': ['densifier']
            },
            dataset_properties={'sparse': True}
        )

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

        self.assertRaisesRegex(ValueError,
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
        np.testing.assert_array_almost_equal(prediction_, prediction)

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
        np.testing.assert_array_almost_equal(prediction_, prediction)

    @unittest.skip("test_check_random_state Not yet Implemented")
    def test_check_random_state(self):
        raise NotImplementedError()

    @unittest.skip("test_validate_input_X Not yet Implemented")
    def test_validate_input_X(self):
        raise NotImplementedError()

    @unittest.skip("test_validate_input_Y Not yet Implemented")
    def test_validate_input_Y(self):
        raise NotImplementedError()

    def test_pipeline_clonability(self):
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='boston')
        auto = SimpleRegressionPipeline()
        auto = auto.fit(X_train, Y_train)
        auto_clone = clone(auto)
        auto_clone_params = auto_clone.get_params()

        # Make sure all keys are copied properly
        for k, v in auto.get_params().items():
            self.assertIn(k, auto_clone_params)

        # Make sure the params getter of estimator are honored
        klass = auto.__class__
        new_object_params = auto.get_params(deep=False)
        for name, param in new_object_params.items():
            new_object_params[name] = clone(param, safe=False)
        new_object = klass(**new_object_params)
        params_set = new_object.get_params(deep=False)

        for name in new_object_params:
            param1 = new_object_params[name]
            param2 = params_set[name]
            self.assertEqual(param1, param2)

    def test_set_params(self):
        pass

    def test_get_params(self):
        pass

    def _test_set_hyperparameter_choice(self, expected_key, implementation, config_dict):
        """
        Given a configuration in config, this procedure makes sure that
        the given implementation, which should be a Choice component, honors
        the type of the object, and any hyperparameter associated to it
        """
        keys_checked = [expected_key]
        implementation_type = config_dict[expected_key]
        expected_type = implementation.get_components()[implementation_type]
        self.assertIsInstance(implementation.choice, expected_type)

        # Are there further hyperparams?
        # A choice component might have attribute requirements that we need to check
        expected_sub_key = expected_key.replace(':__choice__', ':') + implementation_type
        expected_attributes = {}
        for key, value in config_dict.items():
            if key != expected_key and expected_sub_key in key:
                expected_attributes[key.split(':')[-1]] = value
                keys_checked.append(key)
        if expected_attributes:
            attributes = vars(implementation.choice)
            # Cannot check the whole dictionary, just names, as some
            # classes map the text hyperparameter directly to a function!
            for expected_attribute in expected_attributes.keys():
                self.assertIn(expected_attribute, attributes.keys())
        return keys_checked

    def _test_set_hyperparameter_component(self, expected_key, implementation, config_dict):
        """
        Given a configuration in config, this procedure makes sure that
        the given implementation, which should be a autosklearn component, honors
        is created with the desired hyperparameters stated in config_dict
        """
        keys_checked = []
        attributes = vars(implementation)
        expected_attributes = {}
        for key, value in config_dict.items():
            if expected_key in key:
                keys_checked.append(key)
                key = key.replace(expected_key + ':', '')
                if ':' in key:
                    raise ValueError("This utility should only be called with a "
                                     "matching string that produces leaf configurations, "
                                     "that is no further colons are expected, yet key={}"
                                     "".format(
                                            key
                                        )
                                     )
                expected_attributes[key] = value
        # Cannot check the whole dictionary, just names, as some
        # classes map the text hyperparameter directly to a function!
        for expected_attribute in expected_attributes.keys():
            self.assertIn(expected_attribute, attributes.keys())
        return keys_checked

    def test_set_hyperparameters_honors_configuration(self):
        """Makes sure that a given configuration is honored in practice.

        This method tests that the set hyperparameters actually create objects
        that comply with the given configuration. It iterates trough the pipeline to
        make sure we did not miss a step, but also checks at the end that every
        configuration from Config was checked
        """

        all_combinations = list(itertools.product([True, False], repeat=4))
        for sparse, multilabel, signed, multiclass, in all_combinations:
            dataset_properties = {
                'sparse': sparse,
                'multilabel': multilabel,
                'multiclass': multiclass,
                'signed': signed,
            }
            auto = SimpleRegressionPipeline(
                random_state=1,
                dataset_properties=dataset_properties,
            )
            cs = auto.get_hyperparameter_search_space()
            config = cs.sample_configuration()

            # Set hyperparameters takes a given config and translate
            # a config to an actual implementation
            auto.set_hyperparameters(config)
            config_dict = config.get_dictionary()

            # keys to check is our mechanism to ensure that every
            # every config key is checked
            keys_checked = []

            for name, step in auto.named_steps.items():
                if name == 'data_preprocessing':
                    # We have to check both the numerical and categorical
                    to_check = {
                        'numerical_transformer': step.numer_ppl.named_steps,
                        'categorical_transformer': step.categ_ppl.named_steps,
                    }

                    for data_type, pipeline in to_check.items():
                        for sub_name, sub_step in pipeline.items():
                            # If it is a Choice, make sure it is the correct one!
                            if isinstance(sub_step, AutoSklearnChoice):
                                key = "data_preprocessing:{}:{}:__choice__".format(
                                    data_type,
                                    sub_name
                                )
                                keys_checked.extend(
                                    self._test_set_hyperparameter_choice(
                                        key, sub_step, config_dict
                                    )
                                )
                            # If it is a component, make sure it has the correct hyperparams
                            elif isinstance(sub_step, AutoSklearnComponent):
                                keys_checked.extend(
                                    self._test_set_hyperparameter_component(
                                        "data_preprocessing:{}:{}".format(
                                            data_type,
                                            sub_name
                                        ),
                                        sub_step, config_dict
                                    )
                                )
                            else:
                                raise ValueError("New type of pipeline component!")
                elif name == 'balancing':
                    keys_checked.extend(
                        self._test_set_hyperparameter_component(
                            'balancing',
                            step, config_dict
                        )
                    )
                elif name == 'feature_preprocessor':
                    keys_checked.extend(
                        self._test_set_hyperparameter_choice(
                            'feature_preprocessor:__choice__', step, config_dict
                        )
                    )
                elif name == 'regressor':
                    keys_checked.extend(
                        self._test_set_hyperparameter_choice(
                            'regressor:__choice__', step, config_dict
                        )
                    )
                else:
                    raise ValueError("Found another type of step! Need to update this check"
                                     " {}. ".format(name)
                                     )

            # Make sure we checked the whole configuration
            self.assertSetEqual(set(config_dict.keys()), set(keys_checked))
