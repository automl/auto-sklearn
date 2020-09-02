import os
import pickle
import sys
import unittest
import unittest.mock

import joblib
from joblib import cpu_count
import numpy as np
import numpy.ma as npma
import pandas as pd
import sklearn
import sklearn.dummy
import sklearn.datasets

import autosklearn.pipeline.util as putil
import autosklearn.estimators  # noqa F401
from autosklearn.estimators import AutoSklearnEstimator
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import accuracy, f1_macro, mean_squared_error, r2
from autosklearn.automl import AutoMLClassifier
from autosklearn.util.backend import Backend, BackendContext
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

sys.path.append(os.path.dirname(__file__))
from base import Base  # noqa (E402: module level import not at top of file)


class ArrayReturningDummyPredictor(sklearn.dummy.DummyClassifier):
    def __init__(self, test):
        self.arr = test
        self.fitted_ = True

    def predict_proba(self, X, *args, **kwargs):
        return self.arr


class EstimatorTest(Base, unittest.TestCase):
    _multiprocess_can_split_ = True

    # def test_fit_partial_cv(self):
    #
    #     output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_partial_cv')
    #     self._setUp(output)
    #
    #     X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    #     automl = AutoSklearnClassifier(time_left_for_this_task=20,
    #                                    per_run_time_limit=5,
    #                                    tmp_folder=output,
    #                                    output_folder=output,
    #                                    resampling_strategy='partial-cv',
    #                                    ensemble_size=0,
    #                                    delete_tmp_folder_after_terminate=False)
    #     automl.fit(X_train, Y_train)
    #
    #     del automl
    #     self._tearDown(output)

    def test_pSMAC_wrong_arguments(self):
        X = np.zeros((100, 100))
        y = np.zeros((100, ))
        self.assertRaisesRegex(
            ValueError,
            "If shared_mode == True tmp_folder must not "
            "be None.",
            lambda shared_mode:
            AutoSklearnClassifier(
                shared_mode=shared_mode,
            ).fit(X, y),
            shared_mode=True
        )

        self.assertRaisesRegex(
            ValueError,
            "If shared_mode == True output_folder must not "
            "be None.",
            lambda shared_mode, tmp_folder:
            AutoSklearnClassifier(
                shared_mode=shared_mode,
                tmp_folder=tmp_folder,
            ).fit(X, y),
            shared_mode=True,
            tmp_folder='/tmp/duitaredxtvbedb'
        )

    def test_feat_type_wrong_arguments(self):
        cls = AutoSklearnClassifier()
        X = np.zeros((100, 100))
        y = np.zeros((100, ))
        self.assertRaisesRegex(
            ValueError,
            'Array feat_type does not have same number of '
            'variables as X has features. 1 vs 100.',
            cls.fit,
            X=X, y=y, feat_type=[True]
        )

        self.assertRaisesRegex(
            ValueError,
            'Array feat_type must only contain strings.',
            cls.fit,
            X=X, y=y, feat_type=[True]*100
        )

        self.assertRaisesRegex(
            ValueError,
            'Only `Categorical` and `Numerical` are '
            'valid feature types, you passed `Car`',
            cls.fit,
            X=X, y=y, feat_type=['Car']*100
        )

    # Mock AutoSklearnEstimator.fit so the test doesn't actually run fit().
    @unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.fit')
    def test_type_of_target(self, mock_estimator):
        # Test that classifier raises error for illegal target types.
        X = np.array([[1, 2],
                      [2, 3],
                      [3, 4],
                      [4, 5],
                      ])
        # Possible target types
        y_binary = np.array([0, 0, 1, 1])
        y_continuous = np.array([0.1, 1.3, 2.1, 4.0])
        y_multiclass = np.array([0, 1, 2, 0])
        y_multilabel = np.array([[0, 1],
                                 [1, 1],
                                 [1, 0],
                                 [0, 0],
                                 ])
        y_multiclass_multioutput = np.array([[0, 1],
                                             [1, 3],
                                             [2, 2],
                                             [5, 3],
                                             ])
        y_continuous_multioutput = np.array([[0.1, 1.5],
                                             [1.2, 3.5],
                                             [2.7, 2.7],
                                             [5.5, 3.9],
                                             ])

        cls = AutoSklearnClassifier()
        # Illegal target types for classification: continuous,
        # multiclass-multioutput, continuous-multioutput.
        self.assertRaisesRegex(ValueError,
                               "Classification with data of type"
                               " multiclass-multioutput is not supported",
                               cls.fit,
                               X=X,
                               y=y_multiclass_multioutput,
                               )

        self.assertRaisesRegex(ValueError,
                               "Classification with data of type"
                               " continuous is not supported",
                               cls.fit,
                               X=X,
                               y=y_continuous,
                               )

        self.assertRaisesRegex(ValueError,
                               "Classification with data of type"
                               " continuous-multioutput is not supported",
                               cls.fit,
                               X=X,
                               y=y_continuous_multioutput,
                               )

        # Legal target types for classification: binary, multiclass,
        # multilabel-indicator.
        try:
            cls.fit(X, y_binary)
        except ValueError:
            self.fail("cls.fit() raised ValueError while fitting "
                      "binary targets")

        try:
            cls.fit(X, y_multiclass)
        except ValueError:
            self.fail("cls.fit() raised ValueError while fitting "
                      "multiclass targets")

        try:
            cls.fit(X, y_multilabel)
        except ValueError:
            self.fail("cls.fit() raised ValueError while fitting "
                      "multilabel-indicator targets")

        # Test that regressor raises error for illegal target types.
        reg = AutoSklearnRegressor()
        # Illegal target types for regression: multilabel-indicator
        # multiclass-multioutput
        self.assertRaisesRegex(
            ValueError,
            "Regression with data of type"
            " multilabel-indicator is not supported",
            reg.fit,
            X=X,
            y=y_multilabel,
        )

        self.assertRaisesRegex(
            ValueError,
            "Regression with data of type"
            " multiclass-multioutput is not supported",
            reg.fit,
            X=X,
            y=y_multiclass_multioutput,
        )

        # Legal target types: continuous, multiclass,
        # continuous-multioutput,
        # binary
        try:
            reg.fit(X, y_continuous)
        except ValueError:
            self.fail("reg.fit() raised ValueError while fitting "
                      "continuous targets")

        try:
            reg.fit(X, y_multiclass)
        except ValueError:
            self.fail("reg.fit() raised ValueError while fitting "
                      "multiclass targets")

        try:
            reg.fit(X, y_continuous_multioutput)
        except ValueError:
            self.fail("reg.fit() raised ValueError while fitting "
                      "continuous_multioutput targets")

        try:
            reg.fit(X, y_binary)
        except ValueError:
            self.fail("reg.fit() raised ValueError while fitting "
                      "binary targets")

    def test_fit_pSMAC(self):
        tmp = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_pSMAC')
        output = os.path.join(self.test_dir, '..', '.out_estimator_fit_pSMAC')
        self._setUp(tmp)
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')

        # test parallel Classifier to predict classes, not only indices
        Y_train += 1
        Y_test += 1

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            output_folder=output,
            tmp_folder=tmp,
            shared_mode=True,
            seed=1,
            initial_configurations_via_metalearning=0,
            ensemble_size=0,
        )
        automl.fit(X_train, Y_train)
        n_models_fit = len(automl.cv_results_['mean_test_score'])
        cv_results = automl.cv_results_['mean_test_score']

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            output_folder=output,
            tmp_folder=tmp,
            shared_mode=True,
            seed=2,
            initial_configurations_via_metalearning=0,
            ensemble_size=0,
        )
        automl.fit(X_train, Y_train)
        n_models_fit_2 = len(automl.cv_results_['mean_test_score'])

        # Check that the results from the first run were actually read by the
        # second run
        self.assertGreater(n_models_fit_2, n_models_fit)
        for score in cv_results:
            self.assertIn(
                score,
                automl.cv_results_['mean_test_score'],
                msg=str((automl.cv_results_['mean_test_score'], cv_results)),
            )

        # Create a 'dummy model' for the first run, which has an accuracy of
        # more than 99%; it should be in the final ensemble if the ensemble
        # building of the second AutoSklearn classifier works correct
        true_targets_ensemble_path = os.path.join(tmp, '.auto-sklearn',
                                                  'true_targets_ensemble.npy')
        with open(true_targets_ensemble_path, 'rb') as fh:
            true_targets_ensemble = np.load(fh, allow_pickle=True)
        true_targets_ensemble[-1] = 1 if true_targets_ensemble[-1] != 1 else 0
        true_targets_ensemble = true_targets_ensemble.astype(int)
        probas = np.zeros((len(true_targets_ensemble), 2), dtype=float)

        for i, value in enumerate(true_targets_ensemble):
            probas[i, value] = 1.0
        dummy_predictions_path = os.path.join(
            tmp,
            '.auto-sklearn',
            'predictions_ensemble',
            'predictions_ensemble_0_999_0.0.npy',
        )
        with open(dummy_predictions_path, 'wb') as fh:
            np.save(fh, probas)

        probas_test = np.zeros((len(Y_test), 2), dtype=float)
        for i, value in enumerate(Y_test):
            probas_test[i, value - 1] = 1.0

        dummy = ArrayReturningDummyPredictor(probas_test)
        context = BackendContext(tmp, output, False, False, True)
        backend = Backend(context)
        model_path = backend.get_model_path(seed=0, idx=999, budget=0.0)
        backend.save_model(model=dummy, filepath=model_path)

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            output_folder=output,
            tmp_folder=tmp,
            shared_mode=True,
            seed=3,
            initial_configurations_via_metalearning=0,
            ensemble_size=0,
            metric=accuracy,
        )
        automl.fit_ensemble(Y_train, task=BINARY_CLASSIFICATION,
                            precision='32',
                            dataset_name='breast_cancer',
                            ensemble_size=20,
                            ensemble_nbest=50,
                            )

        predictions = automl.predict(X_test)
        score = sklearn.metrics.accuracy_score(Y_test, predictions)

        self.assertEqual(len(os.listdir(os.path.join(tmp, '.auto-sklearn',
                                                     'ensembles'))), 1)
        self.assertGreaterEqual(score, 0.90)
        self.assertEqual(automl._automl[0]._task, BINARY_CLASSIFICATION)

        models = automl._automl[0].models_
        classifier_types = [type(c) for c in models.values()]
        self.assertIn(ArrayReturningDummyPredictor, classifier_types)

        del automl
        self._tearDown(tmp)
        self._tearDown(output)

    def test_cv_results(self):
        # TODO restructure and actually use real SMAC output from a long run
        # to do this unittest!
        tmp = os.path.join(self.test_dir, '..', '.tmp_cv_results')
        output = os.path.join(self.test_dir, '..', '.out_cv_results')
        self._setUp(tmp)
        self._setUp(output)
        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')

        cls = AutoSklearnClassifier(time_left_for_this_task=30,
                                    per_run_time_limit=5,
                                    output_folder=output,
                                    tmp_folder=tmp,
                                    shared_mode=False,
                                    seed=1,
                                    initial_configurations_via_metalearning=0,
                                    ensemble_size=0)
        cls.fit(X_train, Y_train)
        cv_results = cls.cv_results_
        self.assertIsInstance(cv_results, dict)
        self.assertIsInstance(cv_results['mean_test_score'], np.ndarray)
        self.assertIsInstance(cv_results['mean_fit_time'], np.ndarray)
        self.assertIsInstance(cv_results['params'], list)
        self.assertIsInstance(cv_results['rank_test_scores'], np.ndarray)
        self.assertTrue([isinstance(val, npma.MaskedArray) for key, val in
                         cv_results.items() if key.startswith('param_')])
        del cls
        self._tearDown(tmp)
        self._tearDown(output)

    @unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.build_automl')
    @unittest.mock.patch('multiprocessing.Process', autospec=True)
    @unittest.mock.patch('autosklearn.estimators._fit_automl')
    def test_fit_n_jobs(self, _fit_automl_patch, Process_patch, build_automl_patch):
        # Return the process patch on call to __init__
        Process_patch.return_value = Process_patch

        cls = AutoSklearnEstimator()
        cls.fit()
        self.assertEqual(build_automl_patch.call_count, 1)
        self.assertEqual(len(build_automl_patch.call_args[0]), 0)
        self.assertEqual(
            build_automl_patch.call_args[1],
            {
                'seed': 1,
                'shared_mode': False,
                'ensemble_size': 50,
                'initial_configurations_via_metalearning': 25,
                'output_folder': None,
                'tmp_folder': None
            },
        )
        self.assertEqual(Process_patch.call_count, 0)

        cls = AutoSklearnEstimator(n_jobs=5)
        cls.fit()
        # Plus the one from the first call
        self.assertEqual(build_automl_patch.call_count, 6)
        self.assertEqual(len(cls._automl), 5)
        for i in range(1, 6):
            self.assertEqual(len(build_automl_patch.call_args_list[i][0]), 0)
            self.assertEqual(len(build_automl_patch.call_args_list[i][1]), 7)
            # Thee seed is a magic mock so there is nothing to compare here...
            self.assertIn('seed', build_automl_patch.call_args_list[i][1])
            self.assertEqual(
                build_automl_patch.call_args_list[i][1]['shared_mode'],
                True,
            )
            self.assertEqual(
                build_automl_patch.call_args_list[i][1]['ensemble_size'],
                50 if i == 1 else 0,
            )
            self.assertEqual(
                build_automl_patch.call_args_list[i][1][
                    'initial_configurations_via_metalearning'
                ],
                25 if i == 1 else 0,
            )
            if i > 1:
                self.assertEqual(
                    build_automl_patch.call_args_list[i][1][
                        'smac_scenario_args']['initial_incumbent'],
                    'RANDOM',
                )

        self.assertEqual(Process_patch.start.call_count, 4)
        for i in range(2, 6):
            self.assertEqual(
                len(Process_patch.call_args_list[i - 2][1]['kwargs']), 3,
            )
            self.assertFalse(
                Process_patch.call_args_list[i - 2][1]['kwargs']['load_models']
            )
        self.assertEqual(Process_patch.join.call_count, 4)

        self.assertEqual(_fit_automl_patch.call_count, 1)
        self.assertEqual(len(_fit_automl_patch.call_args[0]), 0)
        self.assertEqual(len(_fit_automl_patch.call_args[1]), 3)
        self.assertTrue(_fit_automl_patch.call_args[1]['load_models'])

    def test_fit_n_jobs_2(self):
        tmp = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_pSMAC')
        output = os.path.join(self.test_dir, '..', '.out_estimator_fit_pSMAC')
        self._setUp(tmp)
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')

        # test parallel Classifier to predict classes, not only indices
        Y_train += 1
        Y_test += 1

        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            output_folder=output,
            tmp_folder=tmp,
            seed=1,
            initial_configurations_via_metalearning=0,
            ensemble_size=5,
            n_jobs=2,
            include_estimators=['sgd'],
            include_preprocessors=['no_preprocessing'],
        )
        automl.fit(X_train, Y_train)
        n_runs = len(automl.cv_results_['mean_test_score'])

        predictions_dir = automl._automl[0]._backend._get_prediction_output_dir(
            'ensemble'
        )
        predictions = os.listdir(predictions_dir)
        # two instances of the dummy
        self.assertEqual(n_runs, len(predictions) - 2, msg=str(predictions))

        seeds = set()
        for predictions_file in predictions:
            seeds.add(int(predictions_file.split('.')[0].split('_')[2]))

        self.assertEqual(len(seeds), 2)

        ensemble_dir = automl._automl[0]._backend.get_ensemble_dir()
        ensembles = os.listdir(ensemble_dir)

        seeds = set()
        for ensemble_file in ensembles:
            seeds.add(int(ensemble_file.split('.')[0].split('_')[0]))

        self.assertEqual(len(seeds), 1)

    @unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.build_automl')
    def test_fit_n_jobs_negative(self, build_automl_patch):
        n_cores = cpu_count()
        cls = AutoSklearnEstimator(n_jobs=-1)
        cls.fit()
        self.assertEqual(len(cls._automl), n_cores)

    def test_get_number_of_available_cores(self):
        n_cores = cpu_count()
        self.assertGreaterEqual(n_cores, 1)


class AutoMLClassifierTest(Base, unittest.TestCase):
    @unittest.mock.patch('autosklearn.automl.AutoML.predict')
    def test_multiclass_prediction(self, predict_mock):
        predicted_probabilities = [[0, 0, 0.99], [0, 0.99, 0], [0.99, 0, 0],
                                   [0, 0.99, 0], [0, 0, 0.99]]
        predicted_indexes = [2, 1, 0, 1, 2]
        expected_result = ['c', 'b', 'a', 'b', 'c']

        predict_mock.return_value = np.array(predicted_probabilities)

        classifier = AutoMLClassifier(
            time_left_for_this_task=1,
            per_run_time_limit=1,
            backend=None,
        )
        classifier.InputValidator.validate_target(
            pd.DataFrame(expected_result, dtype='category'),
            is_classification=True,
        )

        actual_result = classifier.predict([None] * len(predicted_indexes))

        np.testing.assert_array_equal(expected_result, actual_result)

    @unittest.mock.patch('autosklearn.automl.AutoML.predict')
    def test_multilabel_prediction(self, predict_mock):
        predicted_probabilities = [[0.99, 0],
                                   [0.99, 0],
                                   [0, 0.99],
                                   [0.99, 0.99],
                                   [0.99, 0.99]]
        predicted_indexes = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [1, 1]])
        expected_result = np.array([[2, 13], [2, 13], [1, 17], [2, 17], [2, 17]])

        predict_mock.return_value = np.array(predicted_probabilities)

        classifier = AutoMLClassifier(
            time_left_for_this_task=1,
            per_run_time_limit=1,
            backend=None,
        )
        classifier.InputValidator.validate_target(
            pd.DataFrame(expected_result, dtype='int64'),
            is_classification=True,
        )

        actual_result = classifier.predict([None] * len(predicted_indexes))

        np.testing.assert_array_equal(expected_result, actual_result)

    def test_can_pickle_classifier(self):
        tmp = os.path.join(self.test_dir, '..', '.tmp_can_pickle')
        output = os.path.join(self.test_dir, '..', '.out_can_pickle')
        self._setUp(tmp)
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                       per_run_time_limit=5,
                                       tmp_folder=tmp,
                                       output_folder=output)
        automl.fit(X_train, Y_train)

        initial_predictions = automl.predict(X_test)
        initial_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                          initial_predictions)
        self.assertGreaterEqual(initial_accuracy, 0.75)

        # Test pickle
        dump_file = os.path.join(output, 'automl.dump.pkl')

        with open(dump_file, 'wb') as f:
            pickle.dump(automl, f)

        with open(dump_file, 'rb') as f:
            restored_automl = pickle.load(f)

        restored_predictions = restored_automl.predict(X_test)
        restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                           restored_predictions)
        self.assertGreaterEqual(restored_accuracy, 0.75)

        self.assertEqual(initial_accuracy, restored_accuracy)

        # Test joblib
        dump_file = os.path.join(output, 'automl.dump.joblib')

        joblib.dump(automl, dump_file)

        restored_automl = joblib.load(dump_file)

        restored_predictions = restored_automl.predict(X_test)
        restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
                                                           restored_predictions)
        self.assertGreaterEqual(restored_accuracy, 0.75)

        self.assertEqual(initial_accuracy, restored_accuracy)

    def test_multilabel(self):
        tmp = os.path.join(self.test_dir, '..', '.tmp_multilabel_fit')
        output = os.path.join(self.test_dir, '..', '.out_multilabel_fit')
        self._setUp(tmp)
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset(
            'iris', make_multilabel=True)
        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                       per_run_time_limit=5,
                                       tmp_folder=tmp,
                                       output_folder=output)

        automl.fit(X_train, Y_train)
        predictions = automl.predict(X_test)
        self.assertEqual(predictions.shape, (50, 3))
        score = f1_macro(Y_test, predictions)
        self.assertGreaterEqual(score, 0.9)
        probs = automl.predict_proba(X_train)
        self.assertAlmostEqual(np.mean(probs), 0.33, places=1)

    def test_binary(self):
        tmp = os.path.join(self.test_dir, '..', '.out_binary_fit')
        output = os.path.join(self.test_dir, '..', '.tmp_binary_fit')
        self._setUp(output)
        self._setUp(tmp)

        X_train, Y_train, X_test, Y_test = putil.get_dataset(
            'iris', make_binary=True)
        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                                       per_run_time_limit=5,
                                       tmp_folder=tmp,
                                       output_folder=output)

        automl.fit(X_train, Y_train, X_test=X_test, y_test=Y_test,
                   dataset_name='binary_test_dataset')
        predictions = automl.predict(X_test)
        self.assertEqual(predictions.shape, (50, ))
        score = accuracy(Y_test, predictions)
        self.assertGreaterEqual(score, 0.9)

        output_files = os.listdir(output)
        self.assertIn('binary_test_dataset_test_1.predict', output_files)

    def test_classification_pandas_support(self):
        X, y = sklearn.datasets.fetch_openml(
            data_id=2,  # cat/num dataset
            return_X_y=True,
            as_frame=True,
        )

        # Drop NAN!!
        X = X.dropna('columns')

        # This test only make sense if input is dataframe
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=5,
            exclude_estimators=['libsvm_svc'],
            seed=5,
        )

        automl.fit(X, y)

        # Make sure that at least better than random.
        # We use same X_train==X_test to test code quality
        self.assertTrue(automl.score(X, y) > 0.555)

        automl.refit(X, y)

        # Make sure that at least better than random.
        # accuracy in sklearn needs valid data
        # It should be 0.555 as the dataset is unbalanced.
        y = automl._automl[0].InputValidator.encode_target(y)
        prediction = automl._automl[0].InputValidator.encode_target(automl.predict(X))
        self.assertTrue(accuracy(y, prediction) > 0.555)


class AutoMLRegressorTest(Base, unittest.TestCase):
    def test_regression(self):
        tmp = os.path.join(self.test_dir, '..', '.tmp_regression_fit')
        output = os.path.join(self.test_dir, '..', '.out_regression_fit')
        self._setUp(tmp)
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('boston')
        automl = AutoSklearnRegressor(time_left_for_this_task=30,
                                      per_run_time_limit=5,
                                      tmp_folder=tmp,
                                      output_folder=output)

        automl.fit(X_train, Y_train)
        predictions = automl.predict(X_test)
        self.assertEqual(predictions.shape, (356,))
        score = mean_squared_error(Y_test, predictions)
        # On average np.sqrt(30) away from the target -> ~5.5 on average
        # Results with select rates drops avg score to a range of -32.40 to -37, on 30 seconds
        # constraint. With more time_left_for_this_task this is no longer an issue
        self.assertGreaterEqual(score, -37)

    def test_cv_regression(self):
        """
        Makes sure that when using a cv strategy, we are able to fit
        a regressor
        """
        tmp = os.path.join(self.test_dir, '..', '.tmp_regression_fit')
        output = os.path.join(self.test_dir, '..', '.out_regression_fit')
        self._setUp(tmp)
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('boston')
        automl = AutoSklearnRegressor(time_left_for_this_task=30,
                                      per_run_time_limit=5,
                                      resampling_strategy='cv',
                                      tmp_folder=tmp,
                                      output_folder=output)

        automl.fit(X_train, Y_train)
        predictions = automl.predict(X_test)
        self.assertEqual(predictions.shape, (356,))
        score = mean_squared_error(Y_test, predictions)
        # On average np.sqrt(30) away from the target -> ~5.5 on average
        # Results with select rates drops avg score to a range of -32.40 to -37, on 30 seconds
        # constraint. With more time_left_for_this_task this is no longer an issue
        self.assertGreaterEqual(score, -37)

        self._tearDown(tmp)
        self._tearDown(output)

    def test_regression_pandas_support(self):
        X, y = sklearn.datasets.fetch_openml(
            data_id=41514,  # diabetes
            return_X_y=True,
            as_frame=True,
        )
        # This test only make sense if input is dataframe
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))
        automl = AutoSklearnRegressor(
            time_left_for_this_task=30,
            per_run_time_limit=5,
        )

        # Make sure we error out because y is not encoded
        automl.fit(X, y)

        # Make sure that at least better than random.
        # We use same X_train==X_test to test code quality
        self.assertTrue(automl.score(X, y) > 0.5)

        automl.refit(X, y)

        # Make sure that at least better than random.
        self.assertTrue(r2(y, automl.predict(X)) > 0.5)


class AutoSklearnClassifierTest(unittest.TestCase):
    # Currently this class only tests that the methods of AutoSklearnClassifier
    # which should return self actually return self.
    def test_classification_methods_returns_self(self):
        X_train, y_train, X_test, y_test = putil.get_dataset('iris')
        automl = AutoSklearnClassifier(time_left_for_this_task=60,
                                       per_run_time_limit=10,
                                       ensemble_size=0,
                                       exclude_preprocessors=['fast_ica'])

        automl_fitted = automl.fit(X_train, y_train)
        self.assertIs(automl, automl_fitted)

        automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
        self.assertIs(automl, automl_ensemble_fitted)

        automl_refitted = automl.refit(X_train.copy(), y_train.copy())
        self.assertIs(automl, automl_refitted)


class AutoSklearnRegressorTest(unittest.TestCase):
    # Currently this class only tests that the methods of AutoSklearnRegressor
    # that should return self actually return self.
    def test_regression_methods_returns_self(self):
        X_train, y_train, X_test, y_test = putil.get_dataset('boston')
        automl = AutoSklearnRegressor(time_left_for_this_task=30,
                                      per_run_time_limit=5,
                                      ensemble_size=0)

        automl_fitted = automl.fit(X_train, y_train)
        self.assertIs(automl, automl_fitted)

        automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
        self.assertIs(automl, automl_ensemble_fitted)

        automl_refitted = automl.refit(X_train.copy(), y_train.copy())
        self.assertIs(automl, automl_refitted)


class AutoSklearn2ClassifierTest(unittest.TestCase):
    # Currently this class only tests that the methods of AutoSklearnClassifier
    # which should return self actually return self.
    def test_classification_methods_returns_self(self):
        X_train, y_train, X_test, y_test = putil.get_dataset('iris')
        automl = AutoSklearn2Classifier(time_left_for_this_task=60, ensemble_size=0,)

        automl_fitted = automl.fit(X_train, y_train)
        self.assertIs(automl, automl_fitted)

        automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
        self.assertIs(automl, automl_ensemble_fitted)

        automl_refitted = automl.refit(X_train.copy(), y_train.copy())
        self.assertIs(automl, automl_refitted)


if __name__ == "__main__":
    unittest.main()
