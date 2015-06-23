__author__ = 'feurerm'

import sys
import traceback
import unittest

import mock
import numpy as np
from scipy.linalg import LinAlgError
import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.svm
from sklearn.utils.testing import assert_array_almost_equal

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.components.preprocessor_base import ParamSklearnPreprocessingAlgorithm
import ParamSklearn.components.classification as classification_components
import ParamSklearn.components.preprocessing as preprocessing_components
from ParamSklearn.util import get_dataset, DENSE, SPARSE, PREDICTIONS


class TestParamSklearnClassifier(unittest.TestCase):
    def test_io_dict(self):
        classifiers = classification_components._classifiers
        for c in classifiers:
            props = classifiers[c].get_properties()
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
            self.assertFalse(props['handles_regression'])
            self.assertIn('handles_classification', props)
            self.assertIn('handles_multiclass', props)
            self.assertIn('handles_multilabel', props)

    def test_find_classifiers(self):
        classifiers = classification_components._classifiers
        self.assertGreaterEqual(len(classifiers), 1)
        for key in classifiers:
            self.assertIn(ParamSklearnClassificationAlgorithm,
                            classifiers[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
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
            self.assertAlmostEqual(0.95999999999999996,
                sklearn.metrics.accuracy_score(predictions, Y_test))
            scores = auto.predict_proba(X_test)

    def test_configurations(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        for i in range(10):
            config = cs.sample_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
            cls = ParamSklearnClassifier(config, random_state=1)
            print config
            try:
                cls.fit(X_train, Y_train)
                X_test_ = X_test.copy()
                predictions = cls.predict(X_test)
                self.assertIsInstance(predictions, np.ndarray)
                predicted_probabiliets = cls.predict_proba(X_test_)
                self.assertIsInstance(predicted_probabiliets, np.ndarray)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                        e.message or \
                    "removed all features" in e.message:
                    continue
                else:
                    print config
                    raise e
            except LinAlgError as e:
                if "not positive definite, even with jitter" in e.message:
                    continue
                else:
                    print config
                    raise e
            except AttributeError as e:
                # Some error in QDA
                if "log" == e.message:
                    continue
                else:
                    print config
                    raise e
            except RuntimeWarning as e:
                if "invalid value encountered in sqrt" in e.message:
                    continue
                elif "divide by zero encountered in divide" in e.message:
                    continue
                else:
                    print config
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.message:
                    continue
                else:
                    print config
                    raise e
            except MemoryError as e:
                continue

    def test_configurations_sparse(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        for i in range(10):
            config = cs.sample_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=True)
            cls = ParamSklearnClassifier(config, random_state=1)
            try:
                cls.fit(X_train, Y_train)
                predictions = cls.predict(X_test)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in \
                        e.message or \
                                "removed all features" in e.message:
                    continue
                else:
                    print config
                    traceback.print_tb(sys.exc_info()[2])
                    raise e
            except LinAlgError as e:
                if "not positive definite, even with jitter" in e.message:
                    continue
                else:
                    print config
                    raise e
            except AttributeError as e:
                # Some error in QDA
                if "log" == e.message:
                    continue
                else:
                    print config
                    raise e
            except RuntimeWarning as e:
                if "invalid value encountered in sqrt" in e.message:
                    continue
                elif "divide by zero encountered in divide" in e.message:
                    continue
                else:
                    print config
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.message:
                    continue
                else:
                    print config
                    raise e

    def test_get_hyperparameter_search_space(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(138, len(hyperparameters))
        # The four parameters which are always active are classifier,
        # preprocessor, imputation strategy and scaling strategy
        self.assertEqual(len(hyperparameters) - 5, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include_estimators=['libsvm_svc'])
        self.assertEqual(cs.get_hyperparameter('classifier'),
            CategoricalHyperparameter('classifier', ['libsvm_svc']))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            exclude_estimators=['libsvm_svc'])
        self.assertNotIn('libsvm_svc', str(cs))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include_preprocessors=['select_percentile_classification'])
        self.assertEqual(cs.get_hyperparameter('preprocessor'),
            CategoricalHyperparameter('preprocessor',
                                      ['select_percentile_classification']))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            exclude_preprocessors=['select_percentile_classification'])
        self.assertNotIn('select_percentile_classification', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include_preprocessors=['nystroem_sampler'])
        self.assertEqual(cs.get_hyperparameter('preprocessor').choices,
                         ['nystroem_sampler'])

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  balancing:strategy, Value: none\n"
            "  classifier, Value: multinomial_nb\n"
            "  imputation:strategy, Value: mean\n"
            "  multinomial_nb:alpha, Value: 1.0\n"
            "  multinomial_nb:fit_prior, Value: True\n"
            "  preprocessor, Value: truncatedSVD\n"
            "  rescaling:strategy, Value: min/max\n"
            "  truncatedSVD:target_dim, Value: 128\n"
            "violates forbidden clause \(Forbidden: preprocessor == "
            "truncatedSVD && Forbidden: classifier == multinomial_nb\)",
                                ParamSklearnClassifier.get_hyperparameter_search_space,
                                include_estimators=['multinomial_nb'],
                                include_preprocessors=['truncatedSVD'],
                                dataset_properties={'sparse':True})

        # It must also be catched that no classifiers which can handle sparse
        #  data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  balancing:strategy, Value: none\n"
            "  classifier, Value: liblinear_svc\n"
            "  imputation:strategy, Value: mean\n"
            "  liblinear_svc:C, Value: 1.0\n"
            "  liblinear_svc:class_weight, Value: None\n"
            "  liblinear_svc:dual, Constant: False\n"
            "  liblinear_svc:fit_intercept, Constant: True\n"
            "  liblinear_svc:intercept_scaling, Constant: 1\n"
            "  liblinear_svc:loss, Value: l2\n"
            "  liblinear_svc:multi_class, Constant: ovr\n"
            "  liblinear_svc:penalty, Value: l2\n"
            "  liblinear_svc:tol, Value: 0.0001\n"
            "  preprocessor, Value: densifier\n"
            "  rescaling:strategy, Value: min/max\n"
            "violates forbidden clause \(Forbidden: classifier == liblinear_svc &&"
            " Forbidden: preprocessor == densifier\)",
                                ParamSklearnClassifier.get_hyperparameter_search_space,
                                include_estimators=['liblinear_svc'],
                                include_preprocessors=['densifier'],
                                dataset_properties={'sparse': True})

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

        # We now have a preprocessing method that handles this case
        #self.assertRaisesRegexp(ValueError,
        #                        "No classifier to build a configuration space "
        #                        "for...", ParamSklearnClassifier.
        #                        get_hyperparameter_search_space,
        #                        dataset_properties={'multilabel': True,
        #                                            'multiclass': True,
        #                                            'sparse': True})

    def test_predict_batched(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        default = cs.get_default_configuration()
        cls = ParamSklearnClassifier(default)

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls._pipeline)
        cls._pipeline = cls_predict
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
        cls_predict = mock.Mock(wraps=cls._pipeline)
        cls._pipeline = cls_predict
        prediction = cls.predict(X_test, batch_size=20)
        self.assertEqual((1647, 2), prediction.shape)
        self.assertEqual(83, cls_predict.predict.call_count)
        assert_array_almost_equal(prediction_, prediction)

    def test_predict_batched_sparse(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        # Densifier + RF is the only combination that easily tests sparse
        # data with multilabel classification!
        config = Configuration(cs,
            values={"balancing:strategy": "none",
                    "classifier": "random_forest",
                    "imputation:strategy": "mean",
                    "preprocessor": "densifier",
                    'random_forest:bootstrap': 'True',
                    'random_forest:criterion': 'gini',
                    'random_forest:max_depth': 'None',
                    'random_forest:min_samples_split': 2,
                    'random_forest:min_samples_leaf': 2,
                    'random_forest:max_features': 0.5,
                    'random_forest:max_leaf_nodes': 'None',
                    'random_forest:n_estimators': 100,
                    "rescaling:strategy": "min/max"})
        cls = ParamSklearnClassifier(config)

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict(X_test_)
        cls_predict = mock.Mock(wraps=cls._pipeline)
        cls._pipeline = cls_predict
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
        cls_predict = mock.Mock(wraps=cls._pipeline)
        cls._pipeline = cls_predict
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
        cls_predict = mock.Mock(wraps=cls._pipeline.steps[-1][1])
        cls._pipeline.steps[-1] = ("estimator", cls_predict)
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
        cls_predict = mock.Mock(wraps=cls._pipeline.steps[-1][1])
        cls._pipeline.steps[-1] = ("estimator", cls_predict)
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

        # Densifier + RF is the only combination that easily tests sparse
        # data with multilabel classification!
        config = Configuration(cs,
                               values={"balancing:strategy": "none",
                                       "classifier": "random_forest",
                                       "imputation:strategy": "mean",
                                       "preprocessor": "densifier",
                                       'random_forest:bootstrap': 'True',
                                       'random_forest:criterion': 'gini',
                                       'random_forest:max_depth': 'None',
                                       'random_forest:min_samples_split': 2,
                                       'random_forest:min_samples_leaf': 2,
                                       'random_forest:max_features': 0.5,
                                       'random_forest:max_leaf_nodes': 'None',
                                       'random_forest:n_estimators': 100,
                                       "rescaling:strategy": "min/max"})

        # Multiclass
        cls = ParamSklearnClassifier(config)
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = mock.Mock(wraps=cls._pipeline.steps[-1][1])
        cls._pipeline.steps[-1] = ("estimator", cls_predict)
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
        cls_predict = mock.Mock(wraps=cls._pipeline.steps[-1][1])
        cls._pipeline.steps[-1] = ("estimator", cls_predict)
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
