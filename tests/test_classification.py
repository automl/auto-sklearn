__author__ = 'feurerm'

import unittest

import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.svm

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter
from HPOlibConfigSpace.random_sampler import RandomSampler

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
            self.assertAlmostEqual(0.62,
                sklearn.metrics.accuracy_score(predictions, Y_test))
            scores = auto.predict_proba(X_test)

    def test_configurations(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        sampler = RandomSampler(cs, 1)
        for i in range(10):
            config = sampler.sample_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
            cls = ParamSklearnClassifier(config, random_state=1)
            try:
                cls.fit(X_train, Y_train)
                predictions = cls.predict(X_test)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in e.message:
                    continue
                else:
                    raise e

    def test_configurations_sparse(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            dataset_properties={'sparse': True})
        sampler = RandomSampler(cs, 1)
        for i in range(10):
            config = sampler.sample_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                           make_sparse=True)
            cls = ParamSklearnClassifier(config, random_state=1)
            try:
                cls.fit(X_train, Y_train)
                predictions = cls.predict(X_test)
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in e.message:
                    continue
                else:
                    raise e

    def test_get_hyperparameter_search_space(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()
        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(90, len(hyperparameters))
        # The four parameters which are always active are classifier,
        # preprocessor, imputation strategy and scaling strategy
        self.assertEqual(len(hyperparameters) - 4, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include_estimators=['libsvm_svc'])
        self.assertEqual(cs.get_hyperparameter('classifier'),
            CategoricalHyperparameter('classifier', ['libsvm_svc']))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            exclude_estimators=['libsvm_svc'])
        self.assertNotIn('libsvm_svc', str(cs))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            include_preprocessors=['pca'])
        self.assertEqual(cs.get_hyperparameter('preprocessor'),
            CategoricalHyperparameter('preprocessor', ['pca']))

        cs = ParamSklearnClassifier.get_hyperparameter_search_space(
            exclude_preprocessors=['pca'])
        self.assertNotIn('pca', str(cs))

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  bagged_multinomial_nb:alpha, Value: 1.000000\n"
            "  bagged_multinomial_nb:fit_prior, Value: True\n"
            "  bagged_multinomial_nb:max_features, Constant: 1.0\n"
            "  bagged_multinomial_nb:max_samples, Constant: 1.0\n"
            "  bagged_multinomial_nb:n_estimators, Constant: 100\n"
            "  classifier, Value: bagged_multinomial_nb\n"
            "  imputation:strategy, Value: mean\n"
            "  preprocessor, Value: truncatedSVD\n"
            "  rescaling:strategy, Value: min/max\n"
            "  truncatedSVD:target_dim, Value: 128\n"
            "violates forbidden clause \(Forbidden: preprocessor == "
            "truncatedSVD && Forbidden: classifier == bagged_multinomial_nb\)",
                                ParamSklearnClassifier.get_hyperparameter_search_space,
                                include_estimators=['bagged_multinomial_nb'],
                                include_preprocessors=['truncatedSVD'],
                                dataset_properties={'sparse':True})

        # It must also be catched that no classifiers which can handle sparse
        #  data are located behind the densifier
        self.assertRaisesRegexp(ValueError, "Configuration:\n"
            "  classifier, Value: liblinear\n"
            "  imputation:strategy, Value: mean\n"
            "  liblinear:C, Value: 1.000000\n"
            "  liblinear:class_weight, Value: None\n"
            "  liblinear:dual, Constant: False\n"
            "  liblinear:fit_intercept, Constant: True\n"
            "  liblinear:intercept_scaling, Constant: 1\n"
            "  liblinear:loss, Value: l2\n"
            "  liblinear:multi_class, Constant: ovr\n"
            "  liblinear:penalty, Value: l2\n"
            "  liblinear:tol, Value: 0.000100\n"
            "  preprocessor, Value: densifier\n"
            "  rescaling:strategy, Value: min/max\n"
            "violates forbidden clause \(Forbidden: classifier == liblinear &&"
            " Forbidden: preprocessor == densifier\)",
                                ParamSklearnClassifier.get_hyperparameter_search_space,
                                include_estimators=['liblinear'],
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
