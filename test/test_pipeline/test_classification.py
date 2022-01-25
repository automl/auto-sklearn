from typing import Any, Dict, Union

import copy
import itertools
import os
import resource
import tempfile
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
from sklearn.utils.validation import check_is_fitted

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm, AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.components.base import AutoSklearnComponent, AutoSklearnChoice, _addons
import autosklearn.pipeline.components.classification as classification_components
import autosklearn.pipeline.components.feature_preprocessing as preprocessing_components
from autosklearn.pipeline.util import get_dataset
from autosklearn.pipeline.constants import \
    DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS, SIGNED_DATA, INPUT

from test.test_pipeline.ignored_warnings import classifier_warnings, ignore_warnings


class DummyClassifier(AutoSklearnClassificationAlgorithm):
    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'AdaBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': False,
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
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class CrashPreprocessor(AutoSklearnPreprocessingAlgorithm):
    def __init__(*args, **kwargs):
        pass

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'AdaBoost Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (INPUT,)}

    def fit(self, X, y):
        raise ValueError("Make sure fit is called")

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


class SimpleClassificationPipelineTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_io_dict(self):
        """Test for the properties of classifier components

        Expects
        -------
        * All required properties are stated in class `get_properties()`
        """
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
        """Test that the classifier components can be found

        Expects
        -------
        * At least two classifier components can be found
        * They inherit from AutoSklearnClassificationAlgorithm
        """
        classifiers = classification_components._classifiers
        self.assertGreaterEqual(len(classifiers), 2)
        for key in classifiers:
            if hasattr(classifiers[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnClassificationAlgorithm, classifiers[key].__bases__)

    def test_find_preprocessors(self):
        """Test that preproccesor components can be found

        Expects
        -------
        * At least 1 preprocessor component can be found
        * The inherit from AutoSklearnPreprocessingAlgorithm
        """
        preprocessors = preprocessing_components._preprocessors
        self.assertGreaterEqual(len(preprocessors),  1)
        for key in preprocessors:
            if hasattr(preprocessors[key], 'get_components'):
                continue
            self.assertIn(AutoSklearnPreprocessingAlgorithm, preprocessors[key].__bases__)

    def test_default_configuration(self):
        """Test that seeded SimpleClassificaitonPipeline returns good results on iris

        Expects
        -------
        * The performance of configuration with fixed seed gets above 96% accuracy on iris
        """
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')

        auto = SimpleClassificationPipeline(random_state=1)

        with ignore_warnings(classifier_warnings):
            auto = auto.fit(X_train, Y_train)

        predictions = auto.predict(X_test)

        acc = sklearn.metrics.accuracy_score(predictions, Y_test)
        self.assertAlmostEqual(0.96, acc)

    def test_default_configuration_multilabel(self):
        """Test that SimpleClassificationPipeline default config returns good results on
        a multilabel version of iris.

        Expects
        -------
        * The performance of a random configuratino gets above 96% on a multilabel
            version of iris
        """
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris', make_multilabel=True)

        classifier = SimpleClassificationPipeline(
            dataset_properties={'multilabel': True},
            random_state=0
        )
        cs = classifier.get_hyperparameter_search_space()

        default = cs.get_default_configuration()
        classifier.set_hyperparameters(default)

        with ignore_warnings(classifier_warnings):
            classifier = classifier.fit(X_train, Y_train)

        predictions = classifier.predict(X_test)

        acc = sklearn.metrics.accuracy_score(predictions, Y_test)
        self.assertAlmostEqual(0.96, acc)

    def test_default_configuration_iterative_fit(self):
        """Test that the SimpleClassificationPipeline default config for random forest
        with no preprocessing can be iteratively fit on iris.

        Expects
        -------
        * Random forest pipeline can be fit iteratively
        * Test that its number of estimators is equal to the iteration count
        """
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')

        classifier = SimpleClassificationPipeline(
            include={
                'classifier': ['random_forest'],
                'feature_preprocessor': ['no_preprocessing']
            },
            random_state=0
        )
        classifier.fit_transformer(X_train, Y_train)

        with ignore_warnings(classifier_warnings):
            for i in range(1, 11):
                classifier.iterative_fit(X_train, Y_train)
                n_estimators = classifier.steps[-1][-1].choice.estimator.n_estimators
                self.assertEqual(n_estimators, i)

    def test_repr(self):
        """Test that the default pipeline can be converted to its representation and
        converted back.

        Expects
        -------
        * The the SimpleClassificationPipeline has a repr
        * This repr can be evaluated back to an instance of SimpleClassificationPipeline
        """
        representation = repr(SimpleClassificationPipeline())
        cls = eval(representation)
        self.assertIsInstance(cls, SimpleClassificationPipeline)

    def test_multilabel(self):
        """Test non-seeded configurations for multi-label data

        Expects
        -------
        * All configurations should fit, predict and predict_proba successfully
        """
        cache = Memory(location=tempfile.gettempdir())
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

        data = {
            'X_train': X[:100, :],
            'Y_train': Y[:100, :],
            'X_test': X[101:, :],
            'Y_test': Y[101:, ]
        }

        pipeline = SimpleClassificationPipeline(dataset_properties={"multilabel": True})
        cs = pipeline.get_hyperparameter_search_space()
        self._test_configurations(configurations_space=cs, dataset=data)

    def test_configurations(self):
        """Tests a non-seeded random set of configurations with default dataset properties

        Expects
        -------
        * All configurations should fit, predict and predict_proba successfully
        """
        cls = SimpleClassificationPipeline()
        cs = cls.get_hyperparameter_search_space()
        self._test_configurations(configurations_space=cs)

    def test_configurations_signed_data(self):
        """Tests a non-seeded random set of configurations with signed data

        Expects
        -------
        * All configurations should fit, predict and predict_proba successfully
        """
        dataset_properties = {'signed': True}

        cls = SimpleClassificationPipeline(dataset_properties=dataset_properties)
        cs = cls.get_hyperparameter_search_space()

        self._test_configurations(configurations_space=cs, dataset_properties=dataset_properties)

    def test_configurations_sparse(self):
        """Tests a non-seeded random set of configurations with sparse data

        Expects
        -------
        * All configurations should fit, predict and predict_proba successfully
        """
        pipeline = SimpleClassificationPipeline(dataset_properties={'sparse': True})
        cs = pipeline.get_hyperparameter_search_space()

        self._test_configurations(configurations_space=cs, make_sparse=True)

    def test_configurations_categorical_data(self):
        """Tests a non-seeded random set of configurations with sparse, mixed data

        Loads specific data from <here>/components/data_preprocessing/dataset.pkl

        Expects
        -------
        * All configurations should fit, predict and predict_proba successfully
        """
        pipeline = SimpleClassificationPipeline(
            dataset_properties={'sparse': False},
            include={
                'feature_preprocessor': ['no_preprocessing'],
                'classifier': ['sgd', 'adaboost']
            }
        )

        cs = pipeline.get_hyperparameter_search_space()

        categorical_columns = [
            True, True, True, False, False, True, True, True, False, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, False, False, False, True, True, True
        ]
        categorical = {
            i: 'categorical' if is_categorical else 'numerical'
            for i, is_categorical in enumerate(categorical_columns)
        }

        here = os.path.dirname(__file__)
        dataset_path = os.path.join(here, "components", "data_preprocessing", "dataset.pkl")

        X = np.loadtxt(dataset_path)
        y = X[:, -1].copy()
        X = X[:, :-1]
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y)

        data = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}

        init_params = {'data_preprocessor:feat_type': categorical}

        self._test_configurations(configurations_space=cs, dataset=data, init_params=init_params)

    @unittest.mock.patch('autosklearn.pipeline.components.data_preprocessing'
                         '.DataPreprocessorChoice.set_hyperparameters')
    def test_categorical_passed_to_one_hot_encoder(self, ohe_mock):
        """Test that the feat_types arg is passed to the OneHotEncoder

        Expects
        -------
        * Construction of SimpleClassificationPipeline to pass init_params correctly
            to the OneHotEncoder

        * Setting the pipeline's hyperparameters after construction also correctly
            sets the init params of the OneHotEncoder
        """

        # Mock the _check_init_params_honored as there is no object created,
        # _check_init_params_honored will fail as a datapreprocessor was never created
        with unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline'
                                 '._check_init_params_honored'):

            # Check through construction
            feat_types = {0: 'categorical', 1: 'numerical'}

            cls = SimpleClassificationPipeline(
                init_params={'data_preprocessor:feat_type': feat_types}
            )

            init_args = ohe_mock.call_args[1]['init_params']
            self.assertEqual(init_args, {'feat_type': feat_types})

            # Check through `set_hyperparameters`
            feat_types = {0: 'categorical', 1: 'categorical', 2: 'numerical'}

            default = cls.get_hyperparameter_search_space().get_default_configuration()
            cls.set_hyperparameters(
                configuration=default,
                init_params={'data_preprocessor:feat_type': feat_types},
            )

            init_args = ohe_mock.call_args[1]['init_params']
            self.assertEqual(init_args, {'feat_type': feat_types})

    def _test_configurations(
        self,
        configurations_space: ConfigurationSpace,
        make_sparse: bool = False,
        dataset: Union[str, Dict[str, Any]] = 'digits',
        init_params: Dict[str, Any] = None,
        dataset_properties: Dict[str, Any] = None,
        n_samples: int = 10,
    ):
        """Tests a configuration space by taking multiple samples and fiting each
        before calling predict and predict_proba.

        Parameters
        ----------
        configurations_space: ConfigurationSpace
            The configuration space to sample from

        make_sparse: bool = False
            Whether to make the dataset sparse or not

        dataset: Union[str, Dict[str, Any]] = 'digits'
            Either a dataset name or a dictionary as below. If given a str, it will
            use `make_sparse` and add NaNs to the dataset.

            {'X_train': ..., 'Y_train': ..., 'X_test': ..., 'y_test': ...}

        init_params: Dict[str, Any] = None
            A dictionary of initial parameters to give to the pipeline.

        dataset_properties: Dict[str, Any]
            A dictionary of properties describing the dataset

        n_samples: int = 10
            How many configurations to sample
        """
        # Use a limit of ~3GiB
        limit = 3072 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        for i in range(n_samples):
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

            config._values.update({
                param: value
                for param, value in restrictions.items()
                if param in config and config[param] is not None
            })

            if isinstance(dataset, str):
                X_train, Y_train, X_test, Y_test = get_dataset(
                    dataset=dataset,
                    make_sparse=make_sparse,
                    add_NaNs=True
                )
            else:
                X_train = dataset['X_train'].copy()
                Y_train = dataset['Y_train'].copy()
                X_test = dataset['X_test'].copy()
                dataset['Y_test'].copy()

            init_params_ = copy.deepcopy(init_params)

            cls = SimpleClassificationPipeline(
                dataset_properties=dataset_properties,
                init_params=init_params_
            )
            cls.set_hyperparameters(config, init_params=init_params_)

            # First make sure that for this configuration, setting the parameters
            # does not mistakenly set the estimator as fitted
            for name, step in cls.named_steps.items():
                with self.assertRaisesRegex(sklearn.exceptions.NotFittedError,
                                            "instance is not fitted yet"):
                    check_is_fitted(step)

            try:
                with ignore_warnings(classifier_warnings):
                    cls.fit(X_train, Y_train)

                # After fit, all components should be tagged as fitted
                # by sklearn. Check is fitted raises an exception if that
                # is not the case
                try:
                    for name, step in cls.named_steps.items():
                        check_is_fitted(step)
                except sklearn.exceptions.NotFittedError:
                    self.fail(f"config={config} raised NotFittedError unexpectedly!")

                cls.predict(X_test.copy())
                cls.predict_proba(X_test)

            except MemoryError:
                continue
            except np.linalg.LinAlgError:
                continue
            except ValueError as e:
                if "Floating-point under-/overflow occurred at epoch" in e.args[0]:
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
                elif 'Internal work array size computation failed' in e.args[0]:
                    continue
                else:
                    e.args += (f"config={config}",)
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
                    e.args += (f"config={config}",)
                    raise e

            except UserWarning as e:
                if "FastICA did not converge" in e.args[0]:
                    continue
                else:
                    e.args += (f"config={config}",)
                    raise e

    def test_get_hyperparameter_search_space(self):
        """Test the configuration space returned by a SimpleClassificationPipeline

        Expects
        -------
        * pipeline returns a configurations space
        * 7 rescaling choices
        * 16 classifier choices
        * 13 features preprocessor choices
        * 168 total hyperparameters
        * (n_hyperparameters - 4) different conditionals for the pipeline
        * 53 forbidden combinations
        """
        pipeline = SimpleClassificationPipeline()
        cs = pipeline.get_hyperparameter_search_space()
        self.assertIsInstance(cs, ConfigurationSpace)

        rescale_param = 'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__'
        n_choices = len(cs.get_hyperparameter(rescale_param).choices)
        self.assertEqual(n_choices, 7)

        n_classifiers = len(cs.get_hyperparameter('classifier:__choice__').choices)
        self.assertEqual(n_classifiers, 16)

        n_preprocessors = len(cs.get_hyperparameter('feature_preprocessor:__choice__').choices)
        self.assertEqual(n_preprocessors, 13)

        hyperparameters = cs.get_hyperparameters()
        self.assertEqual(len(hyperparameters), 168)

        # for hp in sorted([str(h) for h in hyperparameters]):
        #    print hp

        # The four components which are always active are classifier,
        # feature preprocessor, balancing and data preprocessing pipeline.
        conditions = cs.get_conditions()
        self.assertEqual(len(hyperparameters) - 4, len(conditions))

        forbiddens = cs.get_forbiddens()
        self.assertEqual(len(forbiddens), 53)

    def test_get_hyperparameter_search_space_include_exclude_models(self):
        """Test the configuration space when using include and exclude

        Expects
        -------
        * Including a classifier choice has pipeline give back matching choice
        * Excluding a classifier choice means it won't show up in the hyperparameter space
        * Including a feature preprocessor has pipeline give back matching choice
        * Excluding a feature preprocessor means it won't show up in the hyperparameter space
        """
        # include a classifier choice
        pipeline = SimpleClassificationPipeline(include={'classifier': ['libsvm_svc']})
        cs = pipeline.get_hyperparameter_search_space()

        expected = CategoricalHyperparameter('classifier:__choice__', ['libsvm_svc'])
        returned = cs.get_hyperparameter('classifier:__choice__')
        self.assertEqual(returned, expected)

        # exclude a classifier choice
        pipeline = SimpleClassificationPipeline(exclude={'classifier': ['libsvm_svc']})
        cs = pipeline.get_hyperparameter_search_space()
        self.assertNotIn('libsvm_svc', str(cs))

        # include a feature preprocessor
        pipeline = SimpleClassificationPipeline(
            include={'feature_preprocessor': ['select_percentile_classification']}
        )
        cs = pipeline.get_hyperparameter_search_space()

        returned = cs.get_hyperparameter('feature_preprocessor:__choice__')
        expected = CategoricalHyperparameter(
            'feature_preprocessor:__choice__',
            ['select_percentile_classification']
        )
        self.assertEqual(returned, expected)

        # exclude a feature preprocessor
        pipeline = SimpleClassificationPipeline(
            exclude={'feature_preprocessor': ['select_percentile_classification']}
        )
        cs = pipeline.get_hyperparameter_search_space()
        self.assertNotIn('select_percentile_classification', str(cs))

    def test_get_hyperparameter_search_space_preprocessor_contradicts_default_classifier(self):
        """Test that the default classifier gets updated based on the legal feature
        preprocessors that come before.

        Expects
        -------
        * With 'densifier' as only legal feature_preprocessor, 'qda' is default classifier
        * With 'nystroem_sampler' as only legal feature_preprocessor, 'sgd' is default classifier
        """
        pipeline = SimpleClassificationPipeline(
            include={'feature_preprocessor': ['densifier']},
            dataset_properties={'sparse': True}
        )
        cs = pipeline.get_hyperparameter_search_space()

        default_choice = cs.get_hyperparameter('classifier:__choice__').default_value
        self.assertEqual(default_choice, 'qda')

        pipeline = SimpleClassificationPipeline(
            include={'feature_preprocessor': ['nystroem_sampler']}
        )
        cs = pipeline.get_hyperparameter_search_space()

        default_choice = cs.get_hyperparameter('classifier:__choice__').default_value
        self.assertEqual(default_choice, 'sgd')

    def test_get_hyperparameter_search_space_only_forbidden_combinations(self):
        """Test that invalid pipeline configurations raise errors

        Expects
        -------
        * 0 combinations are found with 'multinomial_nb' and 'pca' with 'sparse' data
        * Classifiers that can handle sparse but located behind a 'densifier' should
            raise that no legal default configuration can be found
        """
        with self.assertRaisesRegex(AssertionError, "No valid pipeline found."):
            SimpleClassificationPipeline(
                include={
                    'classifier': ['multinomial_nb'],
                    'feature_preprocessor': ['pca']
                },
                dataset_properties={'sparse': True}
            )

        with self.assertRaisesRegex(ValueError, "Cannot find a legal default configuration."):
            SimpleClassificationPipeline(
                include={
                    'classifier': ['liblinear_svc'],
                    'feature_preprocessor': ['densifier']
                },
                dataset_properties={'sparse': True}
            )

    @unittest.skip("Wait until ConfigSpace is fixed.")
    def test_get_hyperparameter_search_space_dataset_properties(self):
        cs_mc = SimpleClassificationPipeline.get_hyperparameter_search_space(
            dataset_properties={'multiclass': True}
        )
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
        """Test that predict_proba predicts the same as the underlying classifier with
        predict_proba argument `batches`.

        Expects
        -------
        * Should expect the output shape to match that of the digits dataset
        * Should expect a fixed call count each test run
        * Should expect predict_proba with `batches` and predict_proba perform near identically
        """
        cls = SimpleClassificationPipeline(include={'classifier': ['sgd']})

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')

        with ignore_warnings(classifier_warnings):
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
        """Test that predict_proba predicts the same as the underlying classifier with
        predict_proba argument `batches`, with a sparse dataset

        Expects
        -------
        * Should expect the output shape to match that of the digits dataset
        * Should expect a fixed call count each test run
        * Should expect predict_proba with `batches` and predict_proba perform near identically
        """
        cls = SimpleClassificationPipeline(
            dataset_properties={'sparse': True},
            include={'classifier': ['sgd']}
        )

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits', make_sparse=True)
        with ignore_warnings(classifier_warnings):
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
        """Test that predict_proba predicts the same as the underlying classifier with
        predict_proba argument `batches`, for multiclass and multilabel data.

        Expects
        -------
        * Should expect the output shape to match that of the digits dataset
        * Should expect a fixed call count each test run
        * Should expect predict_proba with `batches` and predict_proba perform near identically
        """
        # Multiclass
        cls = SimpleClassificationPipeline(include={'classifier': ['sgd']})
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')

        with ignore_warnings(classifier_warnings):
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

        with ignore_warnings(classifier_warnings):
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
        """Test that predict_proba predicts the same as the underlying classifier with
        predict_proba argument `batches`, for multiclass and multilabel data.

        Expects
        -------
        * Should expect the output shape to match that of the digits dataset
        * Should expect a fixed call count each test run
        * Should expect predict_proba with `batches` and predict_proba perform near identically
        """
        cls = SimpleClassificationPipeline(
            dataset_properties={'sparse': True, 'multiclass': True},
            include={'classifier': ['sgd']}
        )

        # Multiclass
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits', make_sparse=True)
        X_test_ = X_test.copy()

        with ignore_warnings(classifier_warnings):
            cls.fit(X_train, Y_train)

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
            include={'classifier': ['lda']}
        )
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits', make_sparse=True)

        X_test_ = X_test.copy()
        Y_train = np.array([[1 if i != y else 0 for i in range(10)] for y in Y_train])

        with ignore_warnings(classifier_warnings):
            cls.fit(X_train, Y_train)

        prediction_ = cls.predict_proba(X_test_)

        # The object behind the last step in the pipeline
        cls_predict = unittest.mock.Mock(wraps=cls.steps[-1][1].predict_proba)
        cls.steps[-1][-1].predict_proba = cls_predict

        prediction = cls.predict_proba(X_test, batch_size=20)

        self.assertEqual((1647, 10), prediction.shape)
        self.assertEqual(84, cls_predict.call_count)
        np.testing.assert_array_almost_equal(prediction_, prediction)

    def test_pipeline_clonability(self):
        """Test that the pipeline item is clonable with `sklearn.clone`

        Expects
        -------
        * The cloned object has all the same param keys
        * The cloned object can be constructed from theses params
        * The reconstructed clone and the original have the same param values
        """
        X_train, Y_train, X_test, Y_test = get_dataset(dataset='iris')

        auto = SimpleClassificationPipeline()

        with ignore_warnings(classifier_warnings):
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
        """Test that classifiers can be added

        Expects
        -------
        * There should be 0 components initially
        * There should be 1 component after adding a classifier
        * The classifier should be in the search space of the Pipeline after being added
        """
        self.assertEqual(len(classification_components.additional_components.components), 0)
        self.assertEqual(len(_addons['classification'].components), 0)

        classification_components.add_classifier(DummyClassifier)

        self.assertEqual(len(classification_components.additional_components.components), 1)
        self.assertEqual(len(_addons['classification'].components), 1)

        cs = SimpleClassificationPipeline().get_hyperparameter_search_space()
        self.assertIn('DummyClassifier', str(cs))

        del classification_components.additional_components.components['DummyClassifier']

    def test_add_preprocessor(self):
        """Test that preprocessors can be added

        Expects
        -------
        * There should be 0 components initially
        * There should be 1 component after adding a preprocessor
        * The preprocessor should be in the search space of the Pipeline after being added
        """
        self.assertEqual(len(preprocessing_components.additional_components.components), 0)
        self.assertEqual(len(_addons['feature_preprocessing'].components), 0)

        preprocessing_components.add_preprocessor(DummyPreprocessor)

        self.assertEqual(len(preprocessing_components.additional_components.components), 1)
        self.assertEqual(len(_addons['feature_preprocessing'].components), 1)

        cs = SimpleClassificationPipeline().get_hyperparameter_search_space()
        self.assertIn('DummyPreprocessor', str(cs))

        del preprocessing_components.additional_components.components['DummyPreprocessor']

    def _test_set_hyperparameter_choice(self, expected_key, implementation, config_dict):
        """Given a configuration in config, this procedure makes sure that the given
        implementation, which should be a Choice component, honors the type of the
        object, and any hyperparameter associated to it

        TODO: typing
        """
        keys_checked = [expected_key]
        implementation_type = config_dict[expected_key]
        expected_type = implementation.get_components()[implementation_type]
        self.assertIsInstance(implementation.choice, expected_type)

        # Are there further hyperparams?
        # A choice component might have attribute requirements that we need to check
        expected_sub_key = expected_key.replace(':__choice__', ':') + implementation_type
        expected_attributes = {}
        if 'data_preprocessor:__choice__' in expected_key:
            # We have to check both the numerical and categorical
            to_check = {
                'numerical_transformer': implementation.choice.numer_ppl.named_steps,
                'categorical_transformer': implementation.choice.categ_ppl.named_steps,
            }

            for data_type, pipeline in to_check.items():
                for sub_name, sub_step in pipeline.items():
                    # If it is a Choice, make sure it is the correct one!
                    if isinstance(sub_step, AutoSklearnChoice):
                        key = "data_preprocessor:feature_type:{}:{}:__choice__".format(
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
                                "data_preprocessor:feature_type:{}:{}".format(
                                    data_type,
                                    sub_name
                                ),
                                sub_step, config_dict
                            )
                        )
                    else:
                        raise ValueError("New type of pipeline component!")
            return keys_checked

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

        TODO: typing
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
        # self.assertDictContainsSubset(expected_attributes, attributes)
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
        configuration from Config was checked.

        Also considers random_state and ensures pipeline steps correctly recieve
        the right random_state
        """
        random_state = 1
        all_combinations = list(itertools.product([True, False], repeat=4))
        for sparse, multilabel, signed, multiclass, in all_combinations:
            dataset_properties = {
                'sparse': sparse,
                'multilabel': multilabel,
                'multiclass': multiclass,
                'signed': signed,
            }
            cls = SimpleClassificationPipeline(
                random_state=random_state,
                dataset_properties=dataset_properties,
            )
            cs = cls.get_hyperparameter_search_space()
            config = cs.sample_configuration()

            # Set hyperparameters takes a given config and translate
            # a config to an actual implementation
            cls.set_hyperparameters(config)
            config_dict = config.get_dictionary()

            # keys to check is our mechanism to ensure that every
            # every config key is checked
            keys_checked = []

            for name, step in cls.named_steps.items():
                if name == 'data_preprocessor':
                    keys_checked.extend(
                        self._test_set_hyperparameter_choice(
                            'data_preprocessor:__choice__', step, config_dict
                        )
                    )
                    self.assertEqual(step.random_state, random_state)
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
                    self.assertEqual(step.random_state, random_state)
                elif name == 'classifier':
                    keys_checked.extend(
                        self._test_set_hyperparameter_choice(
                            'classifier:__choice__', step, config_dict
                        )
                    )
                    self.assertEqual(step.random_state, random_state)
                else:
                    raise ValueError("Found another type of step! Need to update this check")

            # Make sure we checked the whole configuration
            self.assertSetEqual(set(config_dict.keys()), set(keys_checked))

    def test_fit_instantiates_component(self):
        """Make sure that if a preprocessor is added, it's fit
        method is called"""
        preprocessing_components.add_preprocessor(CrashPreprocessor)

        # We reduce the search space as forbidden clauses prevent to instantiate
        # the user defined preprocessor manually
        cls = SimpleClassificationPipeline(
            include={'classifier': ['random_forest']}
        )
        cs = cls.get_hyperparameter_search_space()
        self.assertIn('CrashPreprocessor', str(cs))
        config = cs.sample_configuration()
        try:
            config['feature_preprocessor:__choice__'] = 'CrashPreprocessor'
        except Exception as e:
            # In case of failure clean up the components and print enough information
            # to clean up with check in the future
            del preprocessing_components.additional_components.components['CrashPreprocessor']
            self.fail("cs={} config={} Exception={}".format(cs, config, e))

        cls.set_hyperparameters(config)

        with self.assertRaisesRegex(ValueError, "Make sure fit is called"):
            with ignore_warnings(classifier_warnings):
                cls.fit(
                    X=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
                    y=np.array([1, 0, 1, 1])
                )

        del preprocessing_components.additional_components.components['CrashPreprocessor']
