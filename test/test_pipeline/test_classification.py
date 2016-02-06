import os
import resource
import sys
import traceback
import unittest

import mock
import numpy as np
import sklearn.datasets
import sklearn.decomposition
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.svm
from sklearn.utils.testing import assert_array_almost_equal

from ConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm, AutoSklearnPreprocessingAlgorithm
import autosklearn.pipeline.components.classification as classification_components
import autosklearn.pipeline.components.feature_preprocessing as preprocessing_components
from autosklearn.pipeline.util import get_dataset
from autosklearn.pipeline.constants import *


class DummyClassifier(AutoSklearnClassificationAlgorithm):
    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'AdaBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class DummyPreprocessor(AutoSklearnPreprocessingAlgorithm):
    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'AdaBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class SimpleClassificationPipelineTest(unittest.TestCase):
    def test_io_dict(self):
        classifiers = classification_components._classifiers
        for c in classifiers:
            if classifiers[c] == classification_components.ClassifierChoice:
                continue
            props = classifiers[c].get_properties()
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
            self.assertFalse(props['handles_regression'])
            self.assertIn('handles_classification', props)
            self.assertIn('handles_multiclass', props)
            self.assertIn('handles_multilabel', props)

    def test_find_classifiers(self):
        classifiers = classification_components._classifiers
        self.assertGreaterEqual(len(classifiers), 2)
        for key in classifiers:
            if hasattr(classifiers[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnClassificationAlgorithm,
                            classifiers[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            if hasattr(preprocessors[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnPreprocessingAlgorithm,
                            preprocessors[key].__bases__)

    def test_default_configuration(self):
        for i in range(2):
            cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
            auto = SimpleClassificationPipeline(default)
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(X_test)
            self.assertAlmostEqual(0.9599999999999995,
                sklearn.metrics.accuracy_score(predictions, Y_test))
            scores = auto.predict_proba(X_test)

    def test_repr(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        representation = repr(SimpleClassificationPipeline(default))
        cls = eval(representation)
        self.assertIsInstance(cls, SimpleClassificationPipeline)

    def test_multilabel(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        dataset_properties = {'multilabel': True}
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(dataset_properties=dataset_properties)

        print(cs)
        cs.seed(5)

        for i in range(50):
            X, Y = sklearn.datasets.\
                    make_multilabel_classification(n_samples=150,
                                                   n_features=20,
                                                   n_classes=5,
                                                   n_labels=2,
                                                   length=50,
                                                   allow_unlabeled=True,
                                                   sparse=False,
                                                   return_indicator=True,
                                                   return_distributions=False,
                                                   random_state=1)
            X_train = X[:100, :]
            Y_train = Y[:100, :]
            X_test = X[101:, :]
            Y_test = Y[101:, ]

            config = cs.sample_configuration()
            config._populate_values()

            if 'classifier:passive_aggressive:n_iter' in config:
                config._values['classifier:passive_aggressive:n_iter'] = 5
            if 'classifier:sgd:n_iter' in config:
                config._values['classifier:sgd:n_iter'] = 5

            cls = SimpleClassificationPipeline(config, random_state=1)
            print(config)
            try:
                cls.fit(X_train, Y_train)
                X_test_ = X_test.copy()
                predictions = cls.predict(X_test)
                self.assertIsInstance(predictions, np.ndarray)
                predicted_probabilities = cls.predict_proba(X_test_)
                [self.assertIsInstance(i, np.ndarray) for i in predicted_probabilities]
            except np.linalg.LinAlgError:
                continue
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                        e.args[0] or \
                        "removed all features" in e.args[0] or \
                        "all features are discarded" in e.args[0]:
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

    def test_configurations(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()

        print(cs)
        cs.seed(1)

        for i in range(10):
            config = cs.sample_configuration()
            config._populate_values()
            if config['classifier:passive_aggressive:n_iter'] is not None:
                config._values['classifier:passive_aggressive:n_iter'] = 5
            if config['classifier:sgd:n_iter'] is not None:
                config._values['classifier:sgd:n_iter'] = 5

            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
            cls = SimpleClassificationPipeline(config, random_state=1)
            print(config)
            try:
                cls.fit(X_train, Y_train)
                X_test_ = X_test.copy()
                predictions = cls.predict(X_test)
                self.assertIsInstance(predictions, np.ndarray)
                predicted_probabiliets = cls.predict_proba(X_test_)
                self.assertIsInstance(predicted_probabiliets, np.ndarray)
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

    def test_configurations_signed_data(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'signed': True})

        print(cs)

        for i in range(10):
            config = cs.sample_configuration()
            config._populate_values()
            if config['classifier:passive_aggressive:n_iter'] is not None:
                config._values['classifier:passive_aggressive:n_iter'] = 5
            if config['classifier:sgd:n_iter'] is not None:
                config._values['classifier:sgd:n_iter'] = 5

            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
            cls = SimpleClassificationPipeline(config, random_state=1)
            print(config)
            try:
                cls.fit(X_train, Y_train)
                X_test_ = X_test.copy()
                predictions = cls.predict(X_test)
                self.assertIsInstance(predictions, np.ndarray)
                predicted_probabiliets = cls.predict_proba(X_test_)
                self.assertIsInstance(predicted_probabiliets, np.ndarray)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                       e.args[0] or \
                       "removed all features" in e.args[0] or \
                                "all features are discarded" in e.args[0]:
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

    def test_configurations_sparse(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        print(cs)
        for i in range(10):
            config = cs.sample_configuration()
            config._populate_values()
            if config['classifier:passive_aggressive:n_iter'] is not None:
                config._values['classifier:passive_aggressive:n_iter'] = 5
            if config['classifier:sgd:n_iter'] is not None:
                config._values['classifier:sgd:n_iter'] = 5

            print(config)
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=True)
            cls = SimpleClassificationPipeline(config, random_state=1)
            try:
                cls.fit(X_train, Y_train)
                predictions = cls.predict(X_test)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                       e.args[0] or \
                        "removed all features" in e.args[0] or \
                                "all features are discarded" in e.args[0]:
                    continue
                else:
                    print(config)
                    traceback.print_tb(sys.exc_info()[2])
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
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.args[0]:
                    continue
                else:
                    print(config)
                    raise e

    def test_configurations_categorical_data(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        print(cs)
        for i in range(10):
            config = cs.sample_configuration()
            config._populate_values()
            if config['classifier:passive_aggressive:n_iter'] is not None:
                config._values['classifier:passive_aggressive:n_iter'] = 5
            if config['classifier:sgd:n_iter'] is not None:
                config._values['classifier:sgd:n_iter'] = 5

            print(config)
            categorical = [True, True, True, False, False, True, True, True,
                           False, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, False,
                           False, False, True, True, True]
            this_directory = os.path.dirname(__file__)
            X = np.loadtxt(os.path.join(this_directory, "components",
                                        "data_preprocessing", "dataset.pkl"))
            y = X[:, -1].copy()
            X = X[:,:-1]
            X_train, X_test, Y_train, Y_test = \
                sklearn.cross_validation.train_test_split(X, y)

            cls = SimpleClassificationPipeline(config, random_state=1,)
            try:
                cls.fit(X_train, Y_train,
                        init_params={'one_hot_encoding:categorical_features': categorical})
                predictions = cls.predict(X_test)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                    e.args[0] or \
                    "removed all features" in e.args[0] or \
                                "all features are discarded" in e.args[0]:
                    continue
                else:
                    print(config)
                    traceback.print_tb(sys.exc_info()[2])
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
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.args[0]:
                    continue
                else:
                    print(config)
                    raise e

    def test_get_hyperparameter_search_space(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()

        self.assertEqual(len(cs.get_hyperparameter(
            'rescaling:__choice__').choices), 4)
        self.assertEqual(len(cs.get_hyperparameter(
            'classifier:__choice__').choices), 16)
        self.assertEqual(len(cs.get_hyperparameter(
            'preprocessor:__choice__').choices), 14)

        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(144, len(hyperparameters))

        #for hp in sorted([str(h) for h in hyperparameters]):
        #    print hp

        # The four parameters which are always active are classifier,
        # preprocessor, imputation strategy and scaling strategy
        self.assertEqual(len(hyperparameters) - 6, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            include={'classifier': ['libsvm_svc']})
        self.assertEqual(cs.get_hyperparameter('classifier:__choice__'),
            CategoricalHyperparameter('classifier:__choice__', ['libsvm_svc']))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            exclude={'classifier': ['libsvm_svc']})
        self.assertNotIn('libsvm_svc', str(cs))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            include={'preprocessor': ['select_percentile_classification']})
        self.assertEqual(cs.get_hyperparameter('preprocessor:__choice__'),
            CategoricalHyperparameter('preprocessor:__choice__',
                                      ['select_percentile_classification']))

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            exclude={'preprocessor': ['select_percentile_classification']})
        self.assertNotIn('select_percentile_classification', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            include={'preprocessor': ['densifier']},
            dataset_properties={'sparse': True})
        self.assertEqual(cs.get_hyperparameter('classifier:__choice__').default,
                         'qda')

        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            include={'preprocessor': ['nystroem_sampler']})
        self.assertEqual(cs.get_hyperparameter('classifier:__choice__').default,
                         'sgd')

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(AssertionError, "No valid pipeline found.",
                                SimpleClassificationPipeline.get_hyperparameter_search_space,
                                include={'classifier': ['multinomial_nb'],
                                         'preprocessor': ['pca']},
                                dataset_properties={'sparse':True})

        # It must also be catched that no classifiers which can handle sparse
        #  data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Cannot find a legal default "
                                            "configuration.",
                                SimpleClassificationPipeline.get_hyperparameter_search_space,
                                include={'classifier': ['liblinear_svc'],
                                         'preprocessor': ['densifier']},
                                dataset_properties={'sparse': True})

    @unittest.skip("Wait until ConfigSpace is fixed.")
    def test_get_hyperparameter_search_space_dataset_properties(self):
        cs_mc = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'multiclass': True})
        self.assertNotIn('bernoulli_nb', str(cs_mc))

        cs_ml = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'multilabel': True})
        self.assertNotIn('k_nearest_neighbors', str(cs_ml))
        self.assertNotIn('liblinear', str(cs_ml))
        self.assertNotIn('libsvm_svc', str(cs_ml))
        self.assertNotIn('sgd', str(cs_ml))

        cs_sp = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        self.assertIn('extra_trees', str(cs_sp))
        self.assertIn('gradient_boosting', str(cs_sp))
        self.assertIn('random_forest', str(cs_sp))

        cs_mc_ml = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'multilabel': True, 'multiclass': True})
        self.assertEqual(cs_ml, cs_mc_ml)

    def test_predict_batched(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        cls = SimpleClassificationPipeline(default)

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_)
        cls.pipeline_ = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((1647,), prediction.shape)
        self.assertEqual(83, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

        # Multilabel
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        Y_train = np.array([(y, 26 - y) for y in Y_train])
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_)
        cls.pipeline_ = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((1647, 2), prediction.shape)
        self.assertEqual(83, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_batched_sparse(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        config = Configuration(cs,
            values={"balancing:strategy": "none",
                    "classifier:__choice__": "random_forest",
                    "imputation:strategy": "mean",
                    "one_hot_encoding:minimum_fraction": 0.01,
                    "one_hot_encoding:use_minimum_fraction": "True",
                    "preprocessor:__choice__": "no_preprocessing",
                    'classifier:random_forest:bootstrap': 'True',
                    'classifier:random_forest:criterion': 'gini',
                    'classifier:random_forest:max_depth': 'None',
                    'classifier:random_forest:min_samples_split': 2,
                    'classifier:random_forest:min_samples_leaf': 2,
                    'classifier:random_forest:max_features': 0.5,
                    'classifier:random_forest:max_leaf_nodes': 'None',
                    'classifier:random_forest:n_estimators': 100,
                    'classifier:random_forest:min_weight_fraction_leaf': 0.0,
                    "rescaling:__choice__": "min/max"})
        cls = SimpleClassificationPipeline(config)

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_)
        cls.pipeline_ = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((1647,), prediction.shape)
        self.assertEqual(83, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

        # Multilabel
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        Y_train = np.array([(y, 26 - y) for y in Y_train])
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_)
        cls.pipeline_ = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((1647, 2), prediction.shape)
        self.assertEqual(83, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_proba_batched(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
        default = cs.get_default_configuration()

        # Multiclass
        cls = SimpleClassificationPipeline(default)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = mock.Mock(wraps=cls.pipeline_.steps[-1][1])
        cls.pipeline_.steps[-1] = ("estimator", cls_predict)
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.predict_proba.call_count)
        assert_array_almost_equal(prediction_, prediction)

        # Multilabel
        cls = SimpleClassificationPipeline(default)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        Y_train_ = np.zeros((Y_train.shape[0], 10))
        for i, y in enumerate(Y_train):
            Y_train_[i][y] = 1
        Y_train = Y_train_
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_.steps[-1][1])
        cls.pipeline_.steps[-1] = ("estimator", cls_predict)
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, ((1647, 10)))
        self.assertEqual(84, cls_predict.predict_proba.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_proba_batched_sparse(self):
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})

        config = Configuration(cs,
                               values={"balancing:strategy": "none",
                                       "classifier:__choice__": "random_forest",
                                       "imputation:strategy": "mean",
                                       "one_hot_encoding:minimum_fraction": 0.01,
                                       "one_hot_encoding:use_minimum_fraction": 'True',
                                       "preprocessor:__choice__": "no_preprocessing",
                                       'classifier:random_forest:bootstrap': 'True',
                                       'classifier:random_forest:criterion': 'gini',
                                       'classifier:random_forest:max_depth': 'None',
                                       'classifier:random_forest:min_samples_split': 2,
                                       'classifier:random_forest:min_samples_leaf': 2,
                                       'classifier:random_forest:min_weight_fraction_leaf': 0.0,
                                       'classifier:random_forest:max_features': 0.5,
                                       'classifier:random_forest:max_leaf_nodes': 'None',
                                       'classifier:random_forest:n_estimators': 100,
                                       "rescaling:__choice__": "min/max"})

        # Multiclass
        cls = SimpleClassificationPipeline(config)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = mock.Mock(wraps=cls.pipeline_.steps[-1][1])
        cls.pipeline_.steps[-1] = ("estimator", cls_predict)
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.predict_proba.call_count)
        assert_array_almost_equal(prediction_, prediction)

        # Multilabel
        cls = SimpleClassificationPipeline(config)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        Y_train_ = np.zeros((Y_train.shape[0], 10))
        for i, y in enumerate(Y_train):
            Y_train_[i][y] = 1
        Y_train = Y_train_
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_.steps[-1][1])
        cls.pipeline_.steps[-1] = ("estimator", cls_predict)
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual(prediction.shape, ((1647, 10)))
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(84, cls_predict.predict_proba.call_count)
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

    def test_add_classifier(self):
        self.assertEqual(len(classification_components._addons.components), 0)
        classification_components.add_classifier(DummyClassifier)
        self.assertEqual(len(classification_components._addons.components), 1)
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
        self.assertIn('DummyClassifier', str(cs))
        del classification_components._addons.components['DummyClassifier']

    def test_add_preprocessor(self):
        self.assertEqual(len(preprocessing_components._addons.components), 0)
        preprocessing_components.add_preprocessor(DummyPreprocessor)
        self.assertEqual(len(preprocessing_components._addons.components), 1)
        cs = SimpleClassificationPipeline.get_hyperparameter_search_space()
        self.assertIn('DummyPreprocessor', str(cs))
        del preprocessing_components._addons.components['DummyPreprocessor']

