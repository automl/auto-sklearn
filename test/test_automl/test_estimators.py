import glob
import os
import pickle
import re
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

import dask
import dask.distributed

import autosklearn.pipeline.util as putil
from autosklearn.ensemble_builder import MODEL_FN_RE
import autosklearn.estimators  # noqa F401
from autosklearn.estimators import AutoSklearnEstimator
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import accuracy, f1_macro, mean_squared_error, r2
from autosklearn.automl import AutoMLClassifier
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from autosklearn.smbo import get_smac_object

sys.path.append(os.path.dirname(__file__))
from base import Base, extract_msg_from_log, count_succeses  # noqa (E402: module level import not at top of file)


def test_fit_n_jobs(tmp_dir, output_dir):

    X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')

    # test parallel Classifier to predict classes, not only indices
    Y_train += 1
    Y_test += 1

    class get_smac_object_wrapper:

        def __call__(self, *args, **kwargs):
            self.n_jobs = kwargs['n_jobs']
            smac = get_smac_object(*args, **kwargs)
            self.dask_n_jobs = smac.solver.tae_runner.n_workers
            self.dask_client_n_jobs = len(
                smac.solver.tae_runner.client.scheduler_info()['workers']
            )
            return smac
    get_smac_object_wrapper_instance = get_smac_object_wrapper()

    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        output_folder=output_dir,
        tmp_folder=tmp_dir,
        seed=1,
        initial_configurations_via_metalearning=0,
        ensemble_size=5,
        n_jobs=2,
        include_estimators=['sgd'],
        include_preprocessors=['no_preprocessing'],
        get_smac_object_callback=get_smac_object_wrapper_instance,
        max_models_on_disc=None,
    )
    automl.fit(X_train, Y_train)

    # Test that the argument is correctly passed to SMAC
    assert getattr(get_smac_object_wrapper_instance, 'n_jobs') == 2
    assert getattr(get_smac_object_wrapper_instance, 'dask_n_jobs') == 2
    assert getattr(get_smac_object_wrapper_instance, 'dask_client_n_jobs') == 2

    available_num_runs = set()
    for run_key, run_value in automl.automl_.runhistory_.data.items():
        if run_value.additional_info is not None and 'num_run' in run_value.additional_info:
            available_num_runs.add(run_value.additional_info['num_run'])
    predictions_dir = automl.automl_._backend._get_prediction_output_dir(
        'ensemble'
    )
    available_predictions = set()
    predictions = os.listdir(predictions_dir)
    seeds = set()
    for prediction in predictions:
        match = re.match(MODEL_FN_RE, prediction.replace("predictions_ensemble", ""))
        if match:
            num_run = int(match.group(2))
            available_predictions.add(num_run)
            seed = int(match.group(1))
            seeds.add(seed)

    done_dir = automl.automl_._backend.get_done_directory()
    dones = os.listdir(done_dir)
    available_dones = set()
    for done in dones:
        match = re.match(r'([0-9]*)_([0-9]*)', done)
        if match:
            num_run = int(match.group(2))
            available_dones.add(num_run)

    # Remove the dummy prediction, it is not part of the runhistory
    available_predictions.remove(1)
    assert available_num_runs.issubset(available_predictions)
    available_dones.remove(1)
    assert available_dones == available_num_runs

    assert len(seeds) == 1

    ensemble_dir = automl.automl_._backend.get_ensemble_dir()
    ensembles = os.listdir(ensemble_dir)

    seeds = set()
    for ensemble_file in ensembles:
        seeds.add(int(ensemble_file.split('.')[0].split('_')[0]))
    assert len(seeds) == 1

    assert count_succeses(automl.cv_results_) > 0



#
#
# class ArrayReturningDummyPredictor(sklearn.dummy.DummyClassifier):
#     def __init__(self, test):
#         self.arr = test
#         self.fitted_ = True
#
#     def predict_proba(self, X, *args, **kwargs):
#         return self.arr
#
#
# class EstimatorTest(Base, unittest.TestCase):
#     _multiprocess_can_split_ = True
#
#     @classmethod
#     def setUpClass(self):
#         self.client = dask.distributed.get_client('127.0.0.1:4567')
#
#     # def test_fit_partial_cv(self):
#     #
#     #     output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_partial_cv')
#     #     self._setUp(output)
#     #
#     #     X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
#     #     automl = AutoSklearnClassifier(time_left_for_this_task=20,
#     #                                    per_run_time_limit=5,
#     #                                    tmp_folder=output,
#     #                                    output_folder=output,
#     #                                    resampling_strategy='partial-cv',
#     #                                    ensemble_size=0,
#     #                                    delete_tmp_folder_after_terminate=False)
#     #     automl.fit(X_train, Y_train)
#     #
#     #     del automl
#     #     self._tearDown(output)
#
#     def test_feat_type_wrong_arguments(self):
#         cls = AutoSklearnClassifier(dask_client=self.client)
#         X = np.zeros((100, 100))
#         y = np.zeros((100, ))
#         self.assertRaisesRegex(
#             ValueError,
#             'Array feat_type does not have same number of '
#             'variables as X has features. 1 vs 100.',
#             cls.fit,
#             X=X, y=y, feat_type=[True]
#         )
#
#         self.assertRaisesRegex(
#             ValueError,
#             'Array feat_type must only contain strings.',
#             cls.fit,
#             X=X, y=y, feat_type=[True]*100
#         )
#
#         self.assertRaisesRegex(
#             ValueError,
#             'Only `Categorical` and `Numerical` are '
#             'valid feature types, you passed `Car`',
#             cls.fit,
#             X=X, y=y, feat_type=['Car']*100
#         )
#         del cls
#
#     # Mock AutoSklearnEstimator.fit so the test doesn't actually run fit().
#     @unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.fit')
#     def test_type_of_target(self, mock_estimator):
#         # Test that classifier raises error for illegal target types.
#         X = np.array([[1, 2],
#                       [2, 3],
#                       [3, 4],
#                       [4, 5],
#                       ])
#         # Possible target types
#         y_binary = np.array([0, 0, 1, 1])
#         y_continuous = np.array([0.1, 1.3, 2.1, 4.0])
#         y_multiclass = np.array([0, 1, 2, 0])
#         y_multilabel = np.array([[0, 1],
#                                  [1, 1],
#                                  [1, 0],
#                                  [0, 0],
#                                  ])
#         y_multiclass_multioutput = np.array([[0, 1],
#                                              [1, 3],
#                                              [2, 2],
#                                              [5, 3],
#                                              ])
#         y_continuous_multioutput = np.array([[0.1, 1.5],
#                                              [1.2, 3.5],
#                                              [2.7, 2.7],
#                                              [5.5, 3.9],
#                                              ])
#
#         cls = AutoSklearnClassifier(dask_client=self.client)
#         # Illegal target types for classification: continuous,
#         # multiclass-multioutput, continuous-multioutput.
#         self.assertRaisesRegex(ValueError,
#                                "Classification with data of type"
#                                " multiclass-multioutput is not supported",
#                                cls.fit,
#                                X=X,
#                                y=y_multiclass_multioutput,
#                                )
#
#         self.assertRaisesRegex(ValueError,
#                                "Classification with data of type"
#                                " continuous is not supported",
#                                cls.fit,
#                                X=X,
#                                y=y_continuous,
#                                )
#
#         self.assertRaisesRegex(ValueError,
#                                "Classification with data of type"
#                                " continuous-multioutput is not supported",
#                                cls.fit,
#                                X=X,
#                                y=y_continuous_multioutput,
#                                )
#
#         # Legal target types for classification: binary, multiclass,
#         # multilabel-indicator.
#         try:
#             cls.fit(X, y_binary)
#         except ValueError:
#             self.fail("cls.fit() raised ValueError while fitting "
#                       "binary targets")
#
#         try:
#             cls.fit(X, y_multiclass)
#         except ValueError:
#             self.fail("cls.fit() raised ValueError while fitting "
#                       "multiclass targets")
#
#         try:
#             cls.fit(X, y_multilabel)
#         except ValueError:
#             self.fail("cls.fit() raised ValueError while fitting "
#                       "multilabel-indicator targets")
#
#         # Test that regressor raises error for illegal target types.
#         reg = AutoSklearnRegressor(dask_client=self.client)
#         # Illegal target types for regression: multilabel-indicator
#         # multiclass-multioutput
#         self.assertRaisesRegex(
#             ValueError,
#             "Regression with data of type"
#             " multilabel-indicator is not supported",
#             reg.fit,
#             X=X,
#             y=y_multilabel,
#         )
#
#         self.assertRaisesRegex(
#             ValueError,
#             "Regression with data of type"
#             " multiclass-multioutput is not supported",
#             reg.fit,
#             X=X,
#             y=y_multiclass_multioutput,
#         )
#
#         # Legal target types: continuous, multiclass,
#         # continuous-multioutput,
#         # binary
#         try:
#             reg.fit(X, y_continuous)
#         except ValueError:
#             self.fail("reg.fit() raised ValueError while fitting "
#                       "continuous targets")
#
#         try:
#             reg.fit(X, y_multiclass)
#         except ValueError:
#             self.fail("reg.fit() raised ValueError while fitting "
#                       "multiclass targets")
#
#         try:
#             reg.fit(X, y_continuous_multioutput)
#         except ValueError:
#             self.fail("reg.fit() raised ValueError while fitting "
#                       "continuous_multioutput targets")
#
#         try:
#             reg.fit(X, y_binary)
#         except ValueError:
#             self.fail("reg.fit() raised ValueError while fitting "
#                       "binary targets")
#
#         # Cleanup
#         del cls
#         del reg
#
#     def test_cv_results(self):
#         # TODO restructure and actually use real SMAC output from a long run
#         # to do this unittest!
#         tmp = os.path.join(self.test_dir, '..', '.tmp_cv_results')
#         output = os.path.join(self.test_dir, '..', '.out_cv_results')
#         self._setUp(tmp)
#         self._setUp(output)
#         X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
#
#         cls = AutoSklearnClassifier(time_left_for_this_task=30,
#                                     per_run_time_limit=5,
#                                     output_folder=output,
#                                     tmp_folder=tmp,
#                                     seed=1,
#                                     initial_configurations_via_metalearning=0,
#                                     dask_client=self.client,
#                                     ensemble_size=0)
#         cls.fit(X_train, Y_train)
#         cv_results = cls.cv_results_
#         self.assertIsInstance(cv_results, dict)
#         self.assertIsInstance(cv_results['mean_test_score'], np.ndarray)
#         self.assertIsInstance(cv_results['mean_fit_time'], np.ndarray)
#         self.assertIsInstance(cv_results['params'], list)
#         self.assertIsInstance(cv_results['rank_test_scores'], np.ndarray)
#         self.assertTrue([isinstance(val, npma.MaskedArray) for key, val in
#                          cv_results.items() if key.startswith('param_')])
#         del cls
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     def test_fit_n_jobs(self):
#         tmp = os.path.join(self.test_dir, '..', '.tmp_estimator_fit_n_jobs')
#         output = os.path.join(self.test_dir, '..', '.out_estimator_fit_n_jobs')
#         self._setUp(tmp)
#         self._setUp(output)
#
#         X_train, Y_train, X_test, Y_test = putil.get_dataset('breast_cancer')
#
#         # test parallel Classifier to predict classes, not only indices
#         Y_train += 1
#         Y_test += 1
#
#         class get_smac_object_wrapper:
#
#             def __call__(self, *args, **kwargs):
#                 self.n_jobs = kwargs['n_jobs']
#                 smac = get_smac_object(*args, **kwargs)
#                 self.dask_n_jobs = smac.solver.tae_runner.n_workers
#                 self.dask_client_n_jobs = len(
#                     smac.solver.tae_runner.client.scheduler_info()['workers']
#                 )
#                 return smac
#         get_smac_object_wrapper_instance = get_smac_object_wrapper()
#
#         automl = AutoSklearnClassifier(
#             time_left_for_this_task=30,
#             per_run_time_limit=5,
#             output_folder=output,
#             tmp_folder=tmp,
#             seed=1,
#             initial_configurations_via_metalearning=0,
#             ensemble_size=5,
#             n_jobs=2,
#             include_estimators=['sgd'],
#             include_preprocessors=['no_preprocessing'],
#             get_smac_object_callback=get_smac_object_wrapper_instance,
#             max_models_on_disc=None,
#             dask_client=self.client,
#         )
#         automl.fit(X_train, Y_train)
#
#         # Test that the argument is correctly passed to SMAC
#         self.assertEqual(getattr(get_smac_object_wrapper_instance, 'n_jobs'), 2)
#         self.assertEqual(getattr(get_smac_object_wrapper_instance, 'dask_n_jobs'), 2)
#         self.assertEqual(getattr(get_smac_object_wrapper_instance, 'dask_client_n_jobs'), 2)
#
#         available_num_runs = set()
#         for run_key, run_value in automl.automl_.runhistory_.data.items():
#             if run_value.additional_info is not None and 'num_run' in run_value.additional_info:
#                 available_num_runs.add(run_value.additional_info['num_run'])
#         predictions_dir = automl.automl_._backend._get_prediction_output_dir(
#             'ensemble'
#         )
#         available_predictions = set()
#         predictions = os.listdir(predictions_dir)
#         seeds = set()
#         for prediction in predictions:
#             match = re.match(MODEL_FN_RE, prediction.replace("predictions_ensemble", ""))
#             if match:
#                 num_run = int(match.group(2))
#                 available_predictions.add(num_run)
#                 seed = int(match.group(1))
#                 seeds.add(seed)
#
#         done_dir = automl.automl_._backend.get_done_directory()
#         dones = os.listdir(done_dir)
#         available_dones = set()
#         for done in dones:
#             match = re.match(r'([0-9]*)_([0-9]*)', done)
#             if match:
#                 num_run = int(match.group(2))
#                 available_dones.add(num_run)
#
#         # Remove the dummy prediction, it is not part of the runhistory
#         available_predictions.remove(1)
#         self.assertTrue(available_num_runs.issubset(available_predictions))
#         available_dones.remove(1)
#         self.assertSetEqual(available_dones, available_num_runs)
#
#         self.assertEqual(len(seeds), 1)
#
#         ensemble_dir = automl.automl_._backend.get_ensemble_dir()
#         ensembles = os.listdir(ensemble_dir)
#
#         seeds = set()
#         for ensemble_file in ensembles:
#             seeds.add(int(ensemble_file.split('.')[0].split('_')[0]))
#         self.assertEqual(len(seeds), 1)
#
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0)
#
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     @unittest.mock.patch('autosklearn.estimators.AutoSklearnEstimator.build_automl')
#     def test_fit_n_jobs_negative(self, build_automl_patch):
#         n_cores = cpu_count()
#         cls = AutoSklearnEstimator(n_jobs=-1)
#         cls.fit()
#         self.assertEqual(cls._n_jobs, n_cores)
#         del cls
#
#     def test_get_number_of_available_cores(self):
#         n_cores = cpu_count()
#         self.assertGreaterEqual(n_cores, 1)
#
#
# class AutoMLClassifierTest(Base, unittest.TestCase):
#
#     @classmethod
#     def setUpClass(self):
#         self.client = dask.distributed.get_client('127.0.0.1:4567')
#
#     @unittest.mock.patch('autosklearn.automl.AutoML.predict')
#     def test_multiclass_prediction(self, predict_mock):
#         predicted_probabilities = [[0, 0, 0.99], [0, 0.99, 0], [0.99, 0, 0],
#                                    [0, 0.99, 0], [0, 0, 0.99]]
#         predicted_indexes = [2, 1, 0, 1, 2]
#         expected_result = ['c', 'b', 'a', 'b', 'c']
#
#         predict_mock.return_value = np.array(predicted_probabilities)
#
#         backend_mock = unittest.mock.Mock()
#         backend_mock.temporary_directory = '/tmp'
#         classifier = AutoMLClassifier(
#             time_left_for_this_task=1,
#             per_run_time_limit=1,
#             backend=backend_mock,
#             dask_client=self.client,
#         )
#         classifier.InputValidator.validate_target(
#             pd.DataFrame(expected_result, dtype='category'),
#             is_classification=True,
#         )
#
#         actual_result = classifier.predict([None] * len(predicted_indexes))
#
#         np.testing.assert_array_equal(expected_result, actual_result)
#
#     @unittest.mock.patch('autosklearn.automl.AutoML.predict')
#     def test_multilabel_prediction(self, predict_mock):
#         predicted_probabilities = [[0.99, 0],
#                                    [0.99, 0],
#                                    [0, 0.99],
#                                    [0.99, 0.99],
#                                    [0.99, 0.99]]
#         predicted_indexes = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [1, 1]])
#         expected_result = np.array([[2, 13], [2, 13], [1, 17], [2, 17], [2, 17]])
#
#         predict_mock.return_value = np.array(predicted_probabilities)
#
#         backend_mock = unittest.mock.Mock()
#         backend_mock.temporary_directory = '/tmp'
#         classifier = AutoMLClassifier(
#             time_left_for_this_task=1,
#             per_run_time_limit=1,
#             backend=backend_mock,
#             dask_client=self.client,
#         )
#         classifier.InputValidator.validate_target(
#             pd.DataFrame(expected_result, dtype='int64'),
#             is_classification=True,
#         )
#
#         actual_result = classifier.predict([None] * len(predicted_indexes))
#
#         np.testing.assert_array_equal(expected_result, actual_result)
#
#     def test_can_pickle_classifier(self):
#         tmp = os.path.join(self.test_dir, '..', '.tmp_can_pickle')
#         output = os.path.join(self.test_dir, '..', '.out_can_pickle')
#         self._setUp(tmp)
#         self._setUp(output)
#
#         X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
#         automl = AutoSklearnClassifier(time_left_for_this_task=30,
#                                        per_run_time_limit=5,
#                                        tmp_folder=tmp,
#                                        dask_client=self.client,
#                                        output_folder=output)
#         automl.fit(X_train, Y_train)
#
#         initial_predictions = automl.predict(X_test)
#         initial_accuracy = sklearn.metrics.accuracy_score(Y_test,
#                                                           initial_predictions)
#         self.assertGreaterEqual(initial_accuracy, 0.75)
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0)
#
#         # Test pickle
#         dump_file = os.path.join(output, 'automl.dump.pkl')
#
#         with open(dump_file, 'wb') as f:
#             pickle.dump(automl, f)
#
#         with open(dump_file, 'rb') as f:
#             restored_automl = pickle.load(f)
#
#         restored_predictions = restored_automl.predict(X_test)
#         restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
#                                                            restored_predictions)
#         self.assertGreaterEqual(restored_accuracy, 0.75)
#
#         self.assertEqual(initial_accuracy, restored_accuracy)
#
#         # Test joblib
#         dump_file = os.path.join(output, 'automl.dump.joblib')
#
#         joblib.dump(automl, dump_file)
#
#         restored_automl = joblib.load(dump_file)
#
#         restored_predictions = restored_automl.predict(X_test)
#         restored_accuracy = sklearn.metrics.accuracy_score(Y_test,
#                                                            restored_predictions)
#         self.assertGreaterEqual(restored_accuracy, 0.75)
#
#         self.assertEqual(initial_accuracy, restored_accuracy)
#
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     def test_multilabel(self):
#         tmp = os.path.join(self.test_dir, '..', '.tmp_multilabel_fit')
#         output = os.path.join(self.test_dir, '..', '.out_multilabel_fit')
#         self._setUp(tmp)
#         self._setUp(output)
#
#         X_train, Y_train, X_test, Y_test = putil.get_dataset(
#             'iris', make_multilabel=True)
#         automl = AutoSklearnClassifier(time_left_for_this_task=30,
#                                        per_run_time_limit=5,
#                                        tmp_folder=tmp,
#                                        dask_client=self.client,
#                                        output_folder=output)
#
#         automl.fit(X_train, Y_train)
#         # Log file path
#         log_file_path = glob.glob(os.path.join(
#             tmp, 'AutoML*.log'))[0]
#         predictions = automl.predict(X_test)
#         self.assertEqual(predictions.shape, (50, 3), extract_msg_from_log(log_file_path))
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0,
#                            extract_msg_from_log(log_file_path))
#         score = f1_macro(Y_test, predictions)
#         self.assertGreaterEqual(score, 0.9, extract_msg_from_log(log_file_path))
#         probs = automl.predict_proba(X_train)
#         self.assertAlmostEqual(np.mean(probs), 0.33, places=1)
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     def test_binary(self):
#         tmp = os.path.join(self.test_dir, '..', '.out_binary_fit')
#         output = os.path.join(self.test_dir, '..', '.tmp_binary_fit')
#         self._setUp(output)
#         self._setUp(tmp)
#
#         X_train, Y_train, X_test, Y_test = putil.get_dataset(
#             'iris', make_binary=True)
#         automl = AutoSklearnClassifier(time_left_for_this_task=30,
#                                        per_run_time_limit=5,
#                                        tmp_folder=tmp,
#                                        dask_client=self.client,
#                                        output_folder=output)
#
#         automl.fit(X_train, Y_train, X_test=X_test, y_test=Y_test,
#                    dataset_name='binary_test_dataset')
#         log_file_path = glob.glob(os.path.join(
#             tmp, 'AutoML*.log'))[0]
#         predictions = automl.predict(X_test)
#         self.assertEqual(predictions.shape, (50, ), extract_msg_from_log(log_file_path))
#         score = accuracy(Y_test, predictions)
#         self.assertGreaterEqual(score, 0.9, extract_msg_from_log(log_file_path))
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0,
#                            extract_msg_from_log(log_file_path))
#
#         output_files = os.listdir(output)
#         self.assertIn('binary_test_dataset_test_1.predict', output_files)
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     def test_classification_pandas_support(self):
#         tmp = os.path.join(self.test_dir, '..', '.out_pd_class_fit')
#         output = os.path.join(self.test_dir, '..', '.tmp_pd_class_fit')
#         self._setUp(output)
#         self._setUp(tmp)
#         X, y = sklearn.datasets.fetch_openml(
#             data_id=2,  # cat/num dataset
#             return_X_y=True,
#             as_frame=True,
#         )
#
#         # Drop NAN!!
#         X = X.dropna('columns')
#
#         # This test only make sense if input is dataframe
#         self.assertTrue(isinstance(X, pd.DataFrame))
#         self.assertTrue(isinstance(y, pd.Series))
#         automl = AutoSklearnClassifier(
#             time_left_for_this_task=30,
#             per_run_time_limit=5,
#             exclude_estimators=['libsvm_svc'],
#             dask_client=self.client,
#             seed=5,
#             tmp_folder=tmp,
#         )
#
#         automl.fit(X, y)
#
#         log_file_path = glob.glob(os.path.join(
#             tmp, 'AutoML*.log'))[0]
#
#         # Make sure that at least better than random.
#         # We use same X_train==X_test to test code quality
#         self.assertTrue(automl.score(X, y) > 0.555, extract_msg_from_log(log_file_path))
#
#         automl.refit(X, y)
#
#         # Make sure that at least better than random.
#         # accuracy in sklearn needs valid data
#         # It should be 0.555 as the dataset is unbalanced.
#         y = automl.automl_.InputValidator.encode_target(y)
#         prediction = automl.automl_.InputValidator.encode_target(automl.predict(X))
#         self.assertTrue(accuracy(y, prediction) > 0.555)
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0)
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#
# class AutoMLRegressorTest(Base, unittest.TestCase):
#
#     @classmethod
#     def setUpClass(self):
#         self.client = dask.distributed.get_client('127.0.0.1:4567')
#
#     def test_regression(self):
#         tmp = os.path.join(self.test_dir, '..', '.tmp_regression_fit')
#         output = os.path.join(self.test_dir, '..', '.out_regression_fit')
#         self._setUp(tmp)
#         self._setUp(output)
#
#         X_train, Y_train, X_test, Y_test = putil.get_dataset('boston')
#         automl = AutoSklearnRegressor(time_left_for_this_task=30,
#                                       per_run_time_limit=5,
#                                       tmp_folder=tmp,
#                                       dask_client=self.client,
#                                       output_folder=output)
#
#         automl.fit(X_train, Y_train)
#
#         # Log file path
#         log_file_path = glob.glob(os.path.join(
#             tmp, 'AutoML*.log'))[0]
#
#         predictions = automl.predict(X_test)
#         self.assertEqual(predictions.shape, (356,))
#         score = mean_squared_error(Y_test, predictions)
#         # On average np.sqrt(30) away from the target -> ~5.5 on average
#         # Results with select rates drops avg score to a range of -32.40 to -37, on 30 seconds
#         # constraint. With more time_left_for_this_task this is no longer an issue
#         self.assertGreaterEqual(score, -37, extract_msg_from_log(log_file_path))
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0)
#
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     def test_cv_regression(self):
#         """
#         Makes sure that when using a cv strategy, we are able to fit
#         a regressor
#         """
#         tmp = os.path.join(self.test_dir, '..', '.tmp_regression_fit_cv')
#         output = os.path.join(self.test_dir, '..', '.out_regression_fit_cv')
#         self._setUp(tmp)
#         self._setUp(output)
#
#         X_train, Y_train, X_test, Y_test = putil.get_dataset('boston', train_size_maximum=300)
#         automl = AutoSklearnRegressor(time_left_for_this_task=60,
#                                       per_run_time_limit=10,
#                                       resampling_strategy='cv',
#                                       tmp_folder=tmp,
#                                       dask_client=self.client,
#                                       output_folder=output)
#
#         automl.fit(X_train, Y_train)
#
#         # Log file path
#         log_file_path = glob.glob(os.path.join(
#             tmp, 'AutoML*.log'))[0]
#
#         predictions = automl.predict(X_test)
#         self.assertEqual(predictions.shape, (206,))
#         score = r2(Y_test, predictions)
#         print(Y_test)
#         print(predictions)
#         self.assertGreaterEqual(score, 0.1, extract_msg_from_log(log_file_path))
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0)
#
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#     def test_regression_pandas_support(self):
#         tmp = os.path.join(self.test_dir, '..', '.tmp_pd_regression')
#         output = os.path.join(self.test_dir, '..', '.out_pd_regression')
#         self._setUp(tmp)
#         self._setUp(output)
#         X, y = sklearn.datasets.fetch_openml(
#             data_id=41514,  # diabetes
#             return_X_y=True,
#             as_frame=True,
#         )
#         # This test only make sense if input is dataframe
#         self.assertTrue(isinstance(X, pd.DataFrame))
#         self.assertTrue(isinstance(y, pd.Series))
#         automl = AutoSklearnRegressor(
#             time_left_for_this_task=40,
#             per_run_time_limit=5,
#             dask_client=self.client,
#             tmp_folder=tmp,
#         )
#
#         # Make sure we error out because y is not encoded
#         automl.fit(X, y)
#
#         log_file_path = glob.glob(os.path.join(
#             tmp, 'AutoML*.log'))[0]
#
#         # Make sure that at least better than random.
#         # We use same X_train==X_test to test code quality
#         self.assertGreaterEqual(automl.score(X, y), 0.5, extract_msg_from_log(log_file_path))
#
#         automl.refit(X, y)
#
#         # Make sure that at least better than random.
#         self.assertTrue(r2(y, automl.predict(X)) > 0.5)
#         self.assertGreater(self._count_succeses(automl.cv_results_), 0)
#         del automl
#         self._tearDown(tmp)
#         self._tearDown(output)
#
#
# class AutoSklearnClassifierTest(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.client = dask.distributed.get_client('127.0.0.1:4567')
#
#     # Currently this class only tests that the methods of AutoSklearnRegressor
#
#     # Currently this class only tests that the methods of AutoSklearnClassifier
#     # which should return self actually return self.
#     def test_classification_methods_returns_self(self):
#         X_train, y_train, X_test, y_test = putil.get_dataset('iris')
#         automl = AutoSklearnClassifier(time_left_for_this_task=60,
#                                        per_run_time_limit=10,
#                                        ensemble_size=0,
#                                        dask_client=self.client,
#                                        exclude_preprocessors=['fast_ica'])
#
#         automl_fitted = automl.fit(X_train, y_train)
#         self.assertIs(automl, automl_fitted)
#
#         automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
#         self.assertIs(automl, automl_ensemble_fitted)
#
#         automl_refitted = automl.refit(X_train.copy(), y_train.copy())
#         self.assertIs(automl, automl_refitted)
#         del automl
#
#
# class AutoSklearnRegressorTest(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.client = dask.distributed.get_client('127.0.0.1:4567')
#
#     # Currently this class only tests that the methods of AutoSklearnRegressor
#     # that should return self actually return self.
#     def test_regression_methods_returns_self(self):
#         X_train, y_train, X_test, y_test = putil.get_dataset('boston')
#         automl = AutoSklearnRegressor(time_left_for_this_task=30,
#                                       per_run_time_limit=5,
#                                       dask_client=self.client,
#                                       ensemble_size=0)
#
#         automl_fitted = automl.fit(X_train, y_train)
#         self.assertIs(automl, automl_fitted)
#
#         automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
#         self.assertIs(automl, automl_ensemble_fitted)
#
#         automl_refitted = automl.refit(X_train.copy(), y_train.copy())
#         self.assertIs(automl, automl_refitted)
#         del automl
#
#
# class AutoSklearn2ClassifierTest(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.client = dask.distributed.get_client('127.0.0.1:4567')
#
#     # Currently this class only tests that the methods of AutoSklearnClassifier
#     # which should return self actually return self.
#     def test_classification_methods_returns_self(self):
#         X_train, y_train, X_test, y_test = putil.get_dataset('iris')
#         automl = AutoSklearn2Classifier(time_left_for_this_task=60, ensemble_size=0,
#                                         dask_client=self.client)
#
#         automl_fitted = automl.fit(X_train, y_train)
#         self.assertIs(automl, automl_fitted)
#
#         automl_ensemble_fitted = automl.fit_ensemble(y_train, ensemble_size=5)
#         self.assertIs(automl, automl_ensemble_fitted)
#
#         automl_refitted = automl.refit(X_train.copy(), y_train.copy())
#         self.assertIs(automl, automl_refitted)
#         del automl
#
#
# if __name__ == "__main__":
#     unittest.main()
