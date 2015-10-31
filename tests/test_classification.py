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

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.components.base import ParamSklearnClassificationAlgorithm
from ParamSklearn.components.base import ParamSklearnPreprocessingAlgorithm
import ParamSklearn.components.classification as classification_components
import ParamSklearn.components.feature_preprocessing as preprocessing_components
from ParamSklearn.util import get_dataset
from ParamSklearn.constants import *


class TestParamSklearnClassifier(unittest.TestCase):
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
            self.assertIn(ParamSklearnClassificationAlgorithm,
                            classifiers[key].__bases__)

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
            cs = ParamSklearnClassifier.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
            auto = ParamSklearnClassifier(default)
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(X_test)
            self.assertAlmostEqual(0.9599999999999995,
                sklearn.metrics.accuracy_score(predictions, Y_test))
            scores = auto.predict_proba(X_test)

    def test_repr(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        representation = repr(ParamSklearnClassifier(default))
        cls = eval(representation)
        self.assertIsInstance(cls, ParamSklearnClassifier)

    def test_multilabel(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        dataset_properties = {'multilabel': True}
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(dataset_properties=dataset_properties)

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

            cls = ParamSklearnClassifier(config, random_state=1)
            print(config)
            try:
                cls.fit(X_train, Y_train)
                X_test_ = X_test.copy()
                predictions = cls.predict(X_test)
                self.assertIsInstance(predictions, np.ndarray)
                predicted_probabilities = cls.predict_proba(X_test_)
                [self.assertIsInstance(i, np.ndarray) for i in predicted_probabilities]
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

        cs = ParamSklearnClassifier.get_hyperparameter_search_space()

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
            cls = ParamSklearnClassifier(config, random_state=1)
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

    def test_configurations_signed_data(self):
        # Use a limit of ~4GiB
        limit = 4000 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
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
            cls = ParamSklearnClassifier(config, random_state=1)
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

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
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
            cls = ParamSklearnClassifier(config, random_state=1)
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

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
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

            cls = ParamSklearnClassifier(config, random_state=1,)
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
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()

        self.assertEqual(len(cs.get_hyperparameter(
            'rescaling:__choice__').choices), 4)
        self.assertEqual(len(cs.get_hyperparameter(
            'classifier:__choice__').choices), 16)
        self.assertEqual(len(cs.get_hyperparameter(
            'preprocessor:__choice__').choices), 14)

        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(145, len(hyperparameters))

        #for hp in sorted([str(h) for h in hyperparameters]):
        #    print hp

        # The four parameters which are always active are classifier,
        # preprocessor, imputation strategy and scaling strategy
        self.assertEqual(len(hyperparameters) - 6, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include={'classifier': ['libsvm_svc']})
        self.assertEqual(cs.get_hyperparameter('classifier:__choice__'),
            CategoricalHyperparameter('classifier:__choice__', ['libsvm_svc']))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            exclude={'classifier': ['libsvm_svc']})
        self.assertNotIn('libsvm_svc', str(cs))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include={'preprocessor': ['select_percentile_classification']})
        self.assertEqual(cs.get_hyperparameter('preprocessor:__choice__'),
            CategoricalHyperparameter('preprocessor:__choice__',
                                      ['select_percentile_classification']))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            exclude={'preprocessor': ['select_percentile_classification']})
        self.assertNotIn('select_percentile_classification', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include={'preprocessor': ['densifier']},
            dataset_properties={'sparse': True})
        self.assertEqual(cs.get_hyperparameter('classifier:__choice__').default,
                         'qda')

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include={'preprocessor': ['nystroem_sampler']})
        self.assertEqual(cs.get_hyperparameter('classifier:__choice__').default,
                         'sgd')

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(AssertionError, "No valid pipeline found.",
                                ParamSklearnClassifier.get_hyperparameter_search_space,
                                include={'classifier': ['multinomial_nb'],
                                         'preprocessor': ['pca']},
                                dataset_properties={'sparse':True})

        # It must also be catched that no classifiers which can handle sparse
        #  data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Cannot find a legal default "
                                            "configuration.",
                                ParamSklearnClassifier.get_hyperparameter_search_space,
                                include={'classifier': ['liblinear_svc'],
                                         'preprocessor': ['densifier']},
                                dataset_properties={'sparse': True})

    @unittest.skip("Wait until HPOlibConfigSpace is fixed.")
    def test_get_hyperparameter_search_space_dataset_properties(self):
        cs_mc = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'multiclass': True})
        self.assertNotIn('bernoulli_nb', str(cs_mc))

        cs_ml = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'multilabel': True})
        self.assertNotIn('k_nearest_neighbors', str(cs_ml))
        self.assertNotIn('liblinear', str(cs_ml))
        self.assertNotIn('libsvm_svc', str(cs_ml))
        self.assertNotIn('sgd', str(cs_ml))

        cs_sp = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        self.assertIn('extra_trees', str(cs_sp))
        self.assertIn('gradient_boosting', str(cs_sp))
        self.assertIn('random_forest', str(cs_sp))

        cs_mc_ml = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'multilabel': True, 'multiclass': True})
        self.assertEqual(cs_ml, cs_mc_ml)

    def test_predict_batched(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        cls = ParamSklearnClassifier(default)

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
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
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
        cls = ParamSklearnClassifier(config)

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
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        default = cs.get_default_configuration()

        # Multiclass
        cls = ParamSklearnClassifier(default)
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
        cls = ParamSklearnClassifier(default)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        Y_train = np.array([(y, 26 - y) for y in Y_train])
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_.steps[-1][1])
        cls.pipeline_.steps[-1] = ("estimator", cls_predict)
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertIsInstance(prediction, list)
        self.assertEqual(2, len(prediction))
        self.assertEqual((1647, 10), prediction[0].shape)
        self.assertEqual((1647, 10), prediction[1].shape)
        self.assertEqual(84, cls_predict.predict_proba.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_proba_batched_sparse(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
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
        cls = ParamSklearnClassifier(config)
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
        cls = ParamSklearnClassifier(config)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        Y_train = np.array([(y, 26 - y) for y in Y_train])
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        cls_predict = mock.Mock(wraps=cls.pipeline_.steps[-1][1])
        cls.pipeline_.steps[-1] = ("estimator", cls_predict)
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertIsInstance(prediction, list)
        self.assertEqual(2, len(prediction))
        self.assertEqual((1647, 10), prediction[0].shape)
        self.assertEqual((1647, 10), prediction[1].shape)
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
