# -*- encoding: utf-8 -*-
from __future__ import print_function
import queue
import multiprocessing
import os
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration
import numpy as np
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit

from autosklearn.evaluation.train_evaluator import TrainEvaluator
from autosklearn.util import backend

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters, \
    get_multiclass_classification_datamanager

N_TEST_RUNS = 3


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class TestTrainEvaluator(BaseEvaluatorTest, unittest.TestCase):
    _multiprocess_can_split_ = True

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_holdout(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_holdout')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss()

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 1.0)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # three calls because of the holdout, the validation and the test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 3)
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                return self.fully_fitted_call_count > 5

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.iteration = 0

            def side_effect(self, *args):
                self.iteration += 1
                return self.losses[self.iteration]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 5)

        for i in range(1, 6):
            duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        self.assertEqual([cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list], [2, 4, 8, 16, 32])
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 15)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 5)
        self.assertEqual(evaluator.model.fit.call_count, 0)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout_interuption(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                if self.fully_fitted_call_count == 3:
                    raise ValueError()
                return self.fully_fitted_call_count > 5

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout_interuption')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.iteration = 0

            def side_effect(self, *args):
                self.iteration += 1
                return self.losses[self.iteration]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        self.assertRaises(ValueError, evaluator.fit_predict_and_loss, iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 2)

        for i in range(1, 3):
            duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 2)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 6)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 2)
        self.assertEqual(evaluator.model.fit.call_count, 0)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout_not_iterative(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = False
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout_not_iterative')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 1)

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertAlmostEqual(result, 1.0)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 0)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 3)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(), random_state=1,
                                n_folds=5, shuffle=True)

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_cv')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss()

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 1.0)
        self.assertEqual(pipeline_mock.fit.call_count, 5)
        # Fifteen calls because of the holdout, the validation and the test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 15)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], D.data['Y_train'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        # The model prior to fitting is saved, this cannot be directly tested
        # because of the way the mock module is used. Instead, we test whether
        # the if block in which model assignment is done is accessed
        self.assertTrue(evaluator._added_empty_model)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_partial_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(), random_state=1,
                                n_folds=5, shuffle=True)

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_partial_cv')
        D = get_binary_classification_datamanager()
        D.name = 'test'

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)

        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.partial_fit_predict_and_loss(1)

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 0)
        self.assertEqual(result, 1.0)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        self.assertEqual(pipeline_mock.predict_proba.call_count, 3)
        # The model prior to fitting is saved, this cannot be directly tested
        # because of the way the mock module is used. Instead, we test whether
        # the if block in which model assignment is done is accessed
        self.assertTrue(evaluator._added_empty_model)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_partial_cv(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(), random_state=1, n_folds=3)

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                return self.fully_fitted_call_count > 5

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_partial_cv')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.iteration = 0

            def side_effect(self, *args):
                self.iteration += 1
                return self.losses[self.iteration]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        evaluator.partial_fit_predict_and_loss(fold=1, iterative=True)
        # No file output here!
        self.assertEqual(evaluator.file_output.call_count, 0)

        for i in range(1, 6):
            duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        self.assertEqual([cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list], [2, 4, 8, 16, 32])
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertFalse(hasattr(evaluator, 'model'))
        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(pipeline_mock.predict_proba.call_count, 15)

#     def test_file_output(self):
#         output_dir = os.path.join(os.getcwd(), '.test_file_output')
#
#         D = get_regression_datamanager()
#         D.name = 'test'
#
#         configuration_space = get_configuration_space(D.info)
#
#         configuration = configuration_space.sample_configuration()
#         backend_api = backend.create(output_dir, output_dir)
#
#         kfold = StratifiedKFold(y=D.data['Y_train'].flatten(),
#                                 n_folds=5, shuffle=True, random_state=1)
#         evaluator = TrainEvaluator(D, backend_api, configuration,
#                                    cv=kfold,
#                                    with_predictions=True,
#                                    all_scoring_functions=True,
#                                    output_y_test=True)
#
#         loss, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
#             evaluator.fit_predict_and_loss()
#         evaluator.file_output(loss, Y_optimization_pred, Y_valid_pred,
#                               Y_test_pred)
#
#         self.assertTrue(os.path.exists(os.path.join(
#             output_dir, '.auto-sklearn', 'true_targets_ensemble.npy')))
#
#         for i in range(5):
#             try:
#                 shutil.rmtree(output_dir)
#             except Exception:
#                 pass
#
#     def test_predict_proba_binary_classification(self):
#         output_dir = os.path.join(os.getcwd(),
#                                   '.test_predict_proba_binary_classification')
#         D = get_binary_classification_datamanager()
#
#         class Dummy2(object):
#
#             def predict_proba(self, y, batch_size=200):
#                 return np.array([[0.1, 0.9]] * 23)
#
#             def fit(self, X, y):
#                 return self
#
#         model = Dummy2()
#
#         configuration_space = get_configuration_space(
#             D.info,
#             include_estimators=['extra_trees'],
#             include_preprocessors=['select_rates'])
#         configuration = configuration_space.sample_configuration()
#
#         evaluator = HoldoutEvaluator(D, output_dir, configuration,
#                                      include={'classifier': ['extra_trees'],
#                                               'preprocessor': ['select_rates']})
#         evaluator.model = model
#         loss, Y_optimization_pred, Y_valid_pred, Y_test_pred = \
#             evaluator.fit_predict_and_loss()
#
#         for i in range(23):
#             self.assertEqual(0.9, Y_optimization_pred[i][1])
#
#         for i in range(5):
#             try:
#                 shutil.rmtree(output_dir)
#             except Exception:
#                 pass
#
#     def test_datasets(self):
#         for getter in get_dataset_getters():
#             testname = '%s_%s' % (os.path.basename(__file__).
#                                   replace('.pyc', '').replace('.py', ''),
#                                   getter.__name__)
#             with self.subTest(testname):
#                 D = getter()
#                 output_directory = os.path.join(os.path.dirname(__file__),
#                                                 '.%s' % testname)
#                 self.output_directory = output_directory
#                 self.output_directories.append(output_directory)
#
#                 err = np.zeros([N_TEST_RUNS])
#                 for i in range(N_TEST_RUNS):
#                     D_ = copy.deepcopy(D)
#                     evaluator = HoldoutEvaluator(D_, self.output_directory, None)
#
#                     err[i] = evaluator.fit_predict_and_loss()[0]
#
#                     self.assertTrue(np.isfinite(err[i]))
#
#                 for i in range(5):
#                     try:
#                         shutil.rmtree(output_directory)
#                     except Exception:
#                         pass
#
#
#
# class FunctionsTest(unittest.TestCase):
#     def setUp(self):
#         self.queue = multiprocessing.Queue()
#         self.configuration = get_configuration_space(
#             {'task': MULTICLASS_CLASSIFICATION,
#              'is_sparse': False}).get_default_configuration()
#         self.data = get_multiclass_classification_datamanager()
#         self.tmp_dir = os.path.join(os.path.dirname(__file__),
#                                     '.test_holdout_functions')
#
#     def tearDown(self):
#         try:
#             shutil.rmtree(self.tmp_dir)
#         except Exception:
#             pass
#
#     def test_eval_holdout(self):
#         backend_api = backend.create(self.tmp_dir, self.tmp_dir)
#         eval_holdout(self.queue, self.configuration, self.data, backend_api,
#                      1, 1, None, True, False, True, None, None, False)
#         info = self.queue.get()
#         self.assertAlmostEqual(info[1], 0.05)
#         self.assertEqual(info[2], 1)
#         self.assertNotIn('bac_metric', info[3])
#
#     def test_eval_holdout_all_loss_functions(self):
#         backend_api = backend.create(self.tmp_dir, self.tmp_dir)
#         eval_holdout(self.queue, self.configuration, self.data, backend_api,
#                      1, 1, None, True, True, True, None, None, False)
#         info = self.queue.get()
#         self.assertIn('f1_metric: 0.0480549199085;pac_metric: 0.122252018407;'
#                       'acc_metric: 0.0454545454545;auc_metric: 0.0;'
#                       'bac_metric: 0.05;duration: ', info[3])
#         self.assertAlmostEqual(info[1], 0.05)
#         self.assertEqual(info[2], 1)
#
#     def test_eval_holdout_on_subset(self):
#         backend_api = backend.create(self.tmp_dir, self.tmp_dir)
#         eval_holdout(self.queue, self.configuration, self.data,
#                      backend_api, 1, 1, 43, True, False, True, None, None,
#                      False)
#         info = self.queue.get()
#         self.assertAlmostEqual(info[1], 0.1)
#         self.assertEqual(info[2], 1)
#
#     def test_eval_holdout_iterative_fit_no_timeout(self):
#         backend_api = backend.create(self.tmp_dir, self.tmp_dir)
#         eval_iterative_holdout(self.queue, self.configuration, self.data,
#                                backend_api, 1, 1, None, True, False, True,
#                                None, None, False)
#         info = self.queue.get()
#         self.assertAlmostEqual(info[1], 0.05)
#         self.assertEqual(info[2], 1)
#
#     def test_eval_holdout_iterative_fit_on_subset_no_timeout(self):
#         backend_api = backend.create(self.tmp_dir, self.tmp_dir)
#         eval_iterative_holdout(self.queue, self.configuration,
#                                self.data, backend_api, 1, 1, 43, True, False,
#                                True, None, None, False)
#
#         info = self.queue.get()
#         self.assertAlmostEqual(info[1], 0.1)
#         self.assertEqual(info[2], 1)
