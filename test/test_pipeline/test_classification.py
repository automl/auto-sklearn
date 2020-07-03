import copy
import os
import resource
import tempfile
import traceback
import unittest
import unittest.mock

from joblib import Memory
import numpy as np

from sklearn.base import clone
import sklearn.datasets
import sklearn.decomposition
import sklearn.model_selection
import sklearn.ensemble
import sklearn.svm

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm, AutoSklearnPreprocessingAlgorithm
import autosklearn.pipeline.components.classification as classification_components
import autosklearn.pipeline.components.feature_preprocessing as preprocessing_components
from autosklearn.pipeline.util import get_dataset
from autosklearn.pipeline.constants import \
    DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS, SIGNED_DATA, INPUT


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
    _multiprocess_can_split_ = True

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
            self.assertIn(AutoSklearnClassificationAlgorithm, classifiers[key].__bases__)

    def test_find_preprocessors(self):
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            if hasattr(preprocessors[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnPreprocessingAlgorithm, preprocessors[key].__bases__)

    def test_default_configuration(self):
        for i in range(2):
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
            auto = SimpleClassificationPipeline()
            auto = auto.fit(X_train, Y_train)
            predictions = auto.predict(X_test)
            self.assertAlmostEqual(0.96, sklearn.metrics.accuracy_score(predictions, Y_test))
            auto.predict_proba(X_test)

    def test_default_configuration_multilabel(self):
        for i in range(2):
            dataset_properties = {'multilabel': True}
            classifier = SimpleClassificationPipeline(
                dataset_properties=dataset_properties)
            cs = classifier.get_hyperparameter_search_space()
            default = cs.get_default_configuration()
            X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris',
                                                           make_multilabel=True)
            classifier.set_hyperparameters(default)
            classifier = classifier.fit(X_train, Y_train)
            predictions = classifier.predict(X_test)
            self.assertAlmostEqual(0.96,
                                   sklearn.metrics.accuracy_score(predictions,
                                                                  Y_test))
            classifier.predict_proba(X_test)

    def test_default_configuration_iterative_fit(self):
        classifier = SimpleClassificationPipeline(
            include={'classifier': ['random_forest'],
                     'feature_preprocessor': ['no_preprocessing']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
        classifier.fit_transformer(X_train, Y_train)
        for i in range(1, 11):
            classifier.iterative_fit(X_train, Y_train)
            self.assertEqual(classifier.steps[-1][-1].choice.estimator.n_estimators,
                             i)

    def test_repr(self):
        representation = repr(SimpleClassificationPipeline())
        cls = eval(representation)
        self.assertIsInstance(cls, SimpleClassificationPipeline)

    def test_multilabel(self):
        cache = Memory(cachedir=tempfile.gettempdir())
        cached_func = cache.cache(
            sklearn.datasets.make_multilabel_classification
        )
        X, Y = cached_func(
            n_samples=150,
            n_features=20,
            n_classes=5,
            n_labels=2,
            length=50,
            allow_unlabeled=True,
            sparse=False,
            return_indicator=True,
            return_distributions=False,
            random_state=1
        )
        X_train = X[:100, :]
        Y_train = Y[:100, :]
        X_test = X[101:, :]
        Y_test = Y[101:, ]

        data = {'X_train': X_train, 'Y_train': Y_train,
                'X_test': X_test, 'Y_test': Y_test}

        dataset_properties = {'multilabel': True}
        cs = SimpleClassificationPipeline(dataset_properties=dataset_properties).\
            get_hyperparameter_search_space()
        self._test_configurations(configurations_space=cs, data=data)

    def test_configurations(self):
        cs = SimpleClassificationPipeline().get_hyperparameter_search_space()

        self._test_configurations(configurations_space=cs)

    def test_configurations_signed_data(self):
        dataset_properties = {'signed': True}
        cs = SimpleClassificationPipeline(dataset_properties=dataset_properties)\
            .get_hyperparameter_search_space()

        self._test_configurations(configurations_space=cs,
                                  dataset_properties=dataset_properties)

    def test_configurations_sparse(self):
        cs = SimpleClassificationPipeline(dataset_properties={'sparse': True}).\
            get_hyperparameter_search_space()

        self._test_configurations(configurations_space=cs, make_sparse=True)

    def test_configurations_categorical_data(self):
        cs = SimpleClassificationPipeline(
            dataset_properties={'sparse': False},
            include={
                'feature_preprocessor': ['no_preprocessing'],
                'classifier': ['sgd', 'adaboost']
            }
        ).get_hyperparameter_search_space()

        categorical = [True, True, True, False, False, True, True, True,
                       False, True, True, True, True, True, True, True,
                       True, True, True, True, True, True, True, True, True,
                       True, True, True, True, True, True, True, False,
                       False, False, True, True, True]
        this_directory = os.path.dirname(__file__)
        X = np.loadtxt(os.path.join(this_directory, "components",
                                    "data_preprocessing", "dataset.pkl"))
        y = X[:, -1].copy()
        X = X[:, :-1]
        X_train, X_test, Y_train, Y_test = \
            sklearn.model_selection.train_test_split(X, y)
        data = {'X_train': X_train, 'Y_train': Y_train,
                'X_test': X_test, 'Y_test': Y_test}

        init_params = {
            'categorical_encoding:one_hot_encoding:categorical_features':
                categorical
        }

        self._test_configurations(configurations_space=cs, make_sparse=True,
                                  data=data, init_params=init_params)

    @unittest.mock.patch('autosklearn.pipeline.components.data_preprocessing'
                         '.data_preprocessing.DataPreprocessor.set_hyperparameters')
    def test_categorical_passed_to_one_hot_encoder(self, ohe_mock):
        cls = SimpleClassificationPipeline(
            init_params={'data_preprocessing:categorical_features': [True, False]}
        )

        self.assertEqual(
            ohe_mock.call_args[1]['init_params'],
            {'categorical_features': [True, False]}
        )
        default = cls.get_hyperparameter_search_space().get_default_configuration()
        cls.set_hyperparameters(
            configuration=default,
            init_params={'data_preprocessing:categorical_features': [True, True, False]},
        )
        self.assertEqual(
            ohe_mock.call_args[1]['init_params'],
            {'categorical_features': [True, True, False]}
        )

    def _test_configurations(self, configurations_space, make_sparse=False,
                             data=None, init_params=None,
                             dataset_properties=None):
        # Use a limit of ~3GiB
        limit = 3072 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        print(configurations_space)

        for i in range(10):
            config = configurations_space.sample_configuration()
            config._populate_values()

            # Restrict configurations which could take too long on travis-ci
            restrictions = {'classifier:passive_aggressive:n_iter': 5,
                            'classifier:sgd:n_iter': 5,
                            'classifier:adaboost:n_estimators': 50,
                            'classifier:adaboost:max_depth': 1,
                            'feature_preprocessor:kernel_pca:n_components': 10,
                            'feature_preprocessor:kitchen_sinks:n_components': 50,
                            'classifier:proj_logit:max_epochs': 1,
                            'classifier:libsvm_svc:degree': 2,
                            'regressor:libsvm_svr:degree': 2,
                            'feature_preprocessor:truncatedSVD:target_dim': 10,
                            'feature_preprocessor:polynomial:degree': 2,
                            'classifier:lda:n_components': 10,
                            'feature_preprocessor:nystroem_sampler:n_components': 50,
                            'feature_preprocessor:feature_agglomeration:n_clusters': 2,
                            'classifier:gradient_boosting:max_leaf_nodes': 64}

            for restrict_parameter in restrictions:
                restrict_to = restrictions[restrict_parameter]
                if restrict_parameter in config and \
                        config[restrict_parameter] is not None:
                    config._values[restrict_parameter] = restrict_to

            print(config)

            if data is None:
                X_train, Y_train, X_test, Y_test = get_dataset(
                    dataset='digits', make_sparse=make_sparse, add_NaNs=True)
            else:
                X_train = data['X_train'].copy()
                Y_train = data['Y_train'].copy()
                X_test = data['X_test'].copy()
                data['Y_test'].copy()

            init_params_ = copy.deepcopy(init_params)
            cls = SimpleClassificationPipeline(random_state=1,
                                               dataset_properties=dataset_properties,
                                               init_params=init_params_,)
            cls.set_hyperparameters(config, init_params=init_params_)
            try:
                cls.fit(X_train, Y_train)
                cls.predict(X_test.copy())
                cls.predict_proba(X_test)
            except MemoryError:
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
                elif "invalid value encountered in multiply" in e.args[0]:
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
                    print(traceback.format_exc())
                    print(config)
                    raise e
            except UserWarning as e:
                if "FastICA did not converge" in e.args[0]:
                    continue
                else:
                    print(traceback.format_exc())
                    print(config)
                    raise e
            except UnboundLocalError as e:
                if "local variable 'raw_predictions_val' referenced before assignment" in e.args[0]:
                    continue
                else:
                    print(traceback.format_exc())
                    print(config)
                    raise e

    def test_get_hyperparameter_search_space(self):
        cs = SimpleClassificationPipeline().get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)
        conditions = cs.get_conditions()

        self.assertEqual(len(cs.get_hyperparameter(
            'data_preprocessing:numerical_transformer:rescaling:__choice__').choices), 6)
        self.assertEqual(len(cs.get_hyperparameter(
            'classifier:__choice__').choices), 15)
        self.assertEqual(len(cs.get_hyperparameter(
            'feature_preprocessor:__choice__').choices), 13)

        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(153, len(hyperparameters))

        # for hp in sorted([str(h) for h in hyperparameters]):
        #    print hp

        # The four components which are always active are classifier,
        # feature preprocessor, balancing and data preprocessing pipeline.
        self.assertEqual(len(hyperparameters) - 7, len(conditions))

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        cs = SimpleClassificationPipeline(include={'classifier': ['libsvm_svc']})\
            .get_hyperparameter_search_space()
        self.assertEqual(
            cs.get_hyperparameter('classifier:__choice__'),
            CategoricalHyperparameter('classifier:__choice__', ['libsvm_svc']),
            )

        cs = SimpleClassificationPipeline(exclude={'classifier': ['libsvm_svc']}).\
            get_hyperparameter_search_space()
        self.assertNotIn('libsvm_svc', str(cs))

        cs = SimpleClassificationPipeline(
            include={'feature_preprocessor': ['select_percentile_classification']}).\
            get_hyperparameter_search_space()
        fpp1 = cs.get_hyperparameter('feature_preprocessor:__choice__')
        fpp2 = CategoricalHyperparameter(
            'feature_preprocessor:__choice__', ['select_percentile_classification'])
        self.assertEqual(fpp1, fpp2)

        cs = SimpleClassificationPipeline(
            exclude={'feature_preprocessor': ['select_percentile_classification']}
        ).get_hyperparameter_search_space()
        self.assertNotIn('select_percentile_classification', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(self):
        cs = SimpleClassificationPipeline(
            include={'feature_preprocessor': ['densifier']},
            dataset_properties={'sparse': True}).\
            get_hyperparameter_search_space()
        self.assertEqual(cs.get_hyperparameter(
            'classifier:__choice__').default_value,
            'qda'
        )

        cs = SimpleClassificationPipeline(
            include={'feature_preprocessor': ['nystroem_sampler']}).\
            get_hyperparameter_search_space()
        self.assertEqual(cs.get_hyperparameter(
            'classifier:__choice__').default_value,
            'sgd'
        )

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        self.assertRaisesRegex(
            AssertionError,
            "No valid pipeline found.",
            SimpleClassificationPipeline,
            include={
                'classifier': ['multinomial_nb'],
                'feature_preprocessor': ['pca']
            },
            dataset_properties={'sparse': True}
        )

        # It must also be catched that no classifiers which can handle sparse
        #  data are located behind the densifier
        self.assertRaisesRegex(
            ValueError,
            "Cannot find a legal default configuration.",
            SimpleClassificationPipeline,
            include={
                'classifier': ['liblinear_svc'],
                'feature_preprocessor': ['densifier']
            },
            dataset_properties={'sparse': True}
        )

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
        cls = SimpleClassificationPipeline(include={'classifier': ['sgd']})

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

    def test_predict_batched_sparse(self):
        cls = SimpleClassificationPipeline(dataset_properties={'sparse': True},
                                           include={'classifier': ['sgd']})

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

    def test_predict_proba_batched(self):
        # Multiclass
        cls = SimpleClassificationPipeline(include={'classifier': ['sgd']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')

        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

        # Multilabel
        cls = SimpleClassificationPipeline(include={'classifier': ['lda']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
        Y_train = np.array(list([(list([1 if i != y else 0 for i in range(10)]))
                                 for y in Y_train]))
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

    def test_predict_proba_batched_sparse(self):

        cls = SimpleClassificationPipeline(
            dataset_properties={'sparse': True, 'multiclass': True},
            include={'classifier': ['sgd']})

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

        # Multilabel
        cls = SimpleClassificationPipeline(
            dataset_properties={'sparse': True, 'multilabel': True},
            include={'classifier': ['lda']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits',
                                                       make_sparse=True)
        Y_train = np.array(list([(list([1 if i != y else 0 for i in range(10)]))
                                 for y in Y_train]))
        cls.fit(X_train, Y_train)
        X_test_ = X_test.copy()
        prediction_ = cls.predict_proba(X_test_)
        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict
        prediction = cls.predict_proba(X_test, batch_size=20)
        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

    def test_pipeline_clonability(self):
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')
        auto = SimpleClassificationPipeline()
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

    def test_add_classifier(self):
        self.assertEqual(len(classification_components._addons.components), 0)
        classification_components.add_classifier(DummyClassifier)
        self.assertEqual(len(classification_components._addons.components), 1)
        cs = SimpleClassificationPipeline().get_hyperparameter_search_space()
        self.assertIn('DummyClassifier', str(cs))
        del classification_components._addons.components['DummyClassifier']

    def test_add_preprocessor(self):
        self.assertEqual(len(preprocessing_components._addons.components), 0)
        preprocessing_components.add_preprocessor(DummyPreprocessor)
        self.assertEqual(len(preprocessing_components._addons.components), 1)
        cs = SimpleClassificationPipeline().get_hyperparameter_search_space()
        self.assertIn('DummyPreprocessor', str(cs))
        del preprocessing_components._addons.components['DummyPreprocessor']
