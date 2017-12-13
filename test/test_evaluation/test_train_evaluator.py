import copy
import json
import queue
import multiprocessing
import os
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration
import numpy as np
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
import sklearn.model_selection
from smac.tae.execute_ta_run import StatusType, TAEAbortException

from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.evaluation import ExecuteTaFuncWithQueue
from autosklearn.evaluation.util import get_last_result
from autosklearn.evaluation.train_evaluator import TrainEvaluator, \
    eval_holdout, eval_iterative_holdout, eval_cv, eval_partial_cv
from autosklearn.util import backend
from autosklearn.util.pipeline import get_configuration_space
from autosklearn.constants import *
from autosklearn.metrics import accuracy, r2, f1_macro

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters, \
    get_multiclass_classification_datamanager


class BackendMock(object):
    def load_datamanager(self):
        return get_multiclass_classification_datamanager()


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class TestTrainEvaluator(BaseEvaluatorTest, unittest.TestCase):
    _multiprocess_can_split_ = True

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_holdout(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        D.name = 'test'

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        output_dir = os.path.join(os.getcwd(), '.test_holdout')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='holdout',
                                   resampling_strategy_args={'train_size': 0.66},
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy,
                                   subsample=50)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss()

        rval = get_last_result(evaluator.queue)
        result = rval['loss']
        self.assertEqual(len(rval), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 0.45833333333333337)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # three calls because of the holdout, the validation and the test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 3)
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 24)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                # Is called twice as often as call to fit because we also check
                # if we need to add a special indicator to show that this is the
                # final call to iterative fit
                return self.fully_fitted_call_count > 10

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='holdout',
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
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
            rval = evaluator.queue.get(timeout=1)
            result = rval['loss']
            self.assertEqual(len(rval), 3)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        self.assertEqual([cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list], [2, 4, 8, 16, 32])
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 15)
        # 1/3 of 69
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 23)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 5)
        self.assertEqual(evaluator.model.fit.call_count, 0)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout_interuption(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                # Is called twice as often as call to fit because we also check
                # if we need to add a special indicator to show that this is the
                # final call to iterative fit
                if self.fully_fitted_call_count == 5:
                    raise ValueError()
                return self.fully_fitted_call_count > 10

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout_interuption')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='holdout-iterative-fit',
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
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
            rval = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(rval['loss'], 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 2)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 6)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 23)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 2)
        self.assertEqual(evaluator.model.fit.call_count, 0)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout_not_iterative(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = False
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout_not_iterative')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='holdout-iterative-fit',
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 1)

        rval = evaluator.queue.get(timeout=1)
        self.assertAlmostEqual(rval['loss'], 0.47826086956521741)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 0)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 3)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 23)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        output_dir = os.path.join(os.getcwd(), '.test_cv')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda : D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 5},
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss()

        rval = get_last_result(evaluator.queue)
        result = rval['loss']
        self.assertEqual(len(rval), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 0.46376811594202894)
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

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        output_dir = os.path.join(os.getcwd(), '.test_partial_cv')
        D = get_binary_classification_datamanager()
        D.name = 'test'

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='partial-cv',
                                   resampling_strategy_args={'folds': 5},
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)

        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.partial_fit_predict_and_loss(1)

        rval = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 0)
        self.assertEqual(rval['loss'], 0.46666666666666667)
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

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                # Is called twice as often as call to fit because we also check
                # if we need to add a special indicator to show that this is the
                # final call to iterative fit
                return self.fully_fitted_call_count > 10

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_partial_cv')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='partial-cv-iterative-fit',
                                   resampling_strategy_args={'folds': 5},
                                   all_scoring_functions=False,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
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
            rval = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(rval['loss'], 1.0 - (0.2 * i))
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

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('os.makedirs')
    def test_file_output(self, makedirs_mock, backend_mock):

        D = get_regression_datamanager()
        D.name = 'test'
        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_mock, queue=queue_,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 5},
                                   all_scoring_functions=True,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)

        backend_mock.get_model_dir.return_value = True
        evaluator.model = 'model'
        evaluator.Y_optimization = D.data['Y_train']
        rval = evaluator.file_output(D.data['Y_train'], D.data['Y_valid'],
                                     D.data['Y_test'])

        self.assertEqual(rval, (None, None))
        self.assertEqual(backend_mock.save_targets_ensemble.call_count, 1)
        self.assertEqual(backend_mock.save_predictions_as_npy.call_count, 3)
        self.assertEqual(makedirs_mock.call_count, 1)
        self.assertEqual(backend_mock.save_model.call_count, 1)

        # Check for not containing NaNs - that the models don't predict nonsense
        # for unseen data
        D.data['Y_valid'][0] = np.NaN
        rval = evaluator.file_output(D.data['Y_train'], D.data['Y_valid'],
                                     D.data['Y_test'])
        self.assertEqual(rval, (1.0, {'error': 'Model predictions for validation '
                                               'set contains NaNs.'}))
        D.data['Y_train'][0] = np.NaN
        rval = evaluator.file_output(D.data['Y_train'], D.data['Y_valid'],
                                     D.data['Y_test'])
        self.assertEqual(rval, (1.0, {'error': 'Model predictions for '
                                               'optimization set contains '
                                               'NaNs.'}))

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_subsample_indices_classification(self, mock, backend_mock):

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()
        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        evaluator = TrainEvaluator(backend_mock, queue_,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 10},
                                   subsample=10,
                                   metric=accuracy)
        train_indices = np.arange(69, dtype=int)
        train_indices1 = evaluator.subsample_indices(train_indices)
        evaluator.subsample = 20
        train_indices2 = evaluator.subsample_indices(train_indices)
        evaluator.subsample = 30
        train_indices3 = evaluator.subsample_indices(train_indices)
        evaluator.subsample = 67
        train_indices4 = evaluator.subsample_indices(train_indices)
        # Common cases
        for ti in train_indices1:
            self.assertIn(ti, train_indices2)
        for ti in train_indices2:
            self.assertIn(ti, train_indices3)
        for ti in train_indices3:
            self.assertIn(ti, train_indices4)

        # Corner cases
        evaluator.subsample = 0
        self.assertRaisesRegex(ValueError, 'The train_size = 0 should be '
                                           'greater or equal to the number '
                                           'of classes = 2',
                               evaluator.subsample_indices, train_indices)
        # With equal or greater it should return a non-shuffled array of indices
        evaluator.subsample = 69
        train_indices5 = evaluator.subsample_indices(train_indices)
        self.assertTrue(np.all(train_indices5 == train_indices))
        evaluator.subsample = 68
        self.assertRaisesRegex(ValueError, 'The test_size = 1 should be greater'
                                           ' or equal to the number of '
                                           'classes = 2',
                               evaluator.subsample_indices, train_indices)

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_subsample_indices_regression(self, mock, backend_mock):

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()
        evaluator = TrainEvaluator(backend_mock, queue_,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 10},
                                   subsample=30,
                                   metric=accuracy)
        train_indices = np.arange(69, dtype=int)
        train_indices3 = evaluator.subsample_indices(train_indices)
        evaluator.subsample = 67
        train_indices4 = evaluator.subsample_indices(train_indices)
        # Common cases
        for ti in train_indices3:
            self.assertIn(ti, train_indices4)

        # Corner cases
        evaluator.subsample = 0
        train_indices5 = evaluator.subsample_indices(train_indices)
        np.testing.assert_allclose(train_indices5, np.array([]))
        # With equal or greater it should return a non-shuffled array of indices
        evaluator.subsample = 69
        train_indices6 = evaluator.subsample_indices(train_indices)
        np.testing.assert_allclose(train_indices6, train_indices)

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_predict_proba_binary_classification(self, mock, backend_mock):
        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        mock.predict_proba.side_effect = lambda y, batch_size: np.array(
            [[0.1, 0.9]] * y.shape[0]
        )
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_mock, queue_,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 10},
                                   output_y_hat_optimization=False,
                                   metric=accuracy)
        evaluator.fit_predict_and_loss()
        Y_optimization_pred = backend_mock.save_predictions_as_npy.call_args_list[0][0][0]

        for i in range(7):
            self.assertEqual(0.9, Y_optimization_pred[i][1])

    @unittest.mock.patch.object(TrainEvaluator, 'file_output')
    @unittest.mock.patch.object(TrainEvaluator, '_partial_fit_and_predict')
    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_fit_predict_and_loss_additional_run_info(
        self, mock, backend_mock, _partial_fit_and_predict_mock,
            file_output_mock,
    ):
        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        mock.side_effect = lambda **kwargs: mock
        _partial_fit_and_predict_mock.return_value = (
            [[0.1, 0.9]] * 23, [[0.1, 0.9]] * 7, [[0.1, 0.9]] * 7, {'a': 5}
        )
        file_output_mock.return_value = None, None

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(
            backend_mock, queue_,
            configuration=configuration,
            resampling_strategy='holdout',
            output_y_hat_optimization=False,
            metric=accuracy,
        )
        evaluator.Y_targets[0] = [1] * 23
        evaluator.fit_predict_and_loss(iterative=False)

        class SideEffect(object):
            def __init__(self):
                self.n_call = 0
            def __call__(self, *args, **kwargs):
                if self.n_call == 0:
                    self.n_call += 1
                    return (
                        [[0.1, 0.9]] * 35,
                        [[0.1, 0.9]] * 7,
                        [[0.1, 0.9]] * 7,
                        {'a': 5}
                    )
                else:
                    return (
                        [[0.1, 0.9]] * 34,
                        [[0.1, 0.9]] * 7,
                        [[0.1, 0.9]] * 7,
                        {'a': 5}
                    )
        _partial_fit_and_predict_mock.side_effect = SideEffect()
        evaluator = TrainEvaluator(
            backend_mock, queue_,
            configuration=configuration,
            resampling_strategy='cv',
            resampling_strategy_args={'folds': 2},
            output_y_hat_optimization=False,
            metric=accuracy,
        )
        evaluator.Y_targets[0] = [1] * 35
        evaluator.Y_targets[1] = [1] * 34

        self.assertRaises(
            TAEAbortException,
            evaluator.fit_predict_and_loss,
            iterative=False
        )

    def test_get_results(self):
        backend_mock = unittest.mock.Mock(spec=backend.Backend)
        backend_mock.get_model_dir.return_value = 'dutirapbdxvltcrpbdlcatepdeau'
        queue_ = multiprocessing.Queue()
        for i in range(5):
            queue_.put((i * 1, 1 - (i * 0.2), 0, "", StatusType.SUCCESS))
        result = get_last_result(queue_)
        self.assertEqual(result[0], 4)
        self.assertAlmostEqual(result[1], 0.2)

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)

            with self.subTest(testname):
                backend_mock = unittest.mock.Mock(spec=backend.Backend)
                backend_mock.get_model_dir.return_value = 'dutirapbdxvltcrpbdlcatepdeau'
                D = getter()
                D_ = copy.deepcopy(D)
                y = D.data['Y_train']
                if len(y.shape) == 2 and y.shape[1] == 1:
                    D_.data['Y_train'] = y.flatten()
                backend_mock.load_datamanager.return_value = D_
                queue_ = multiprocessing.Queue()
                metric_lookup = {MULTILABEL_CLASSIFICATION: f1_macro,
                                 BINARY_CLASSIFICATION: accuracy,
                                 MULTICLASS_CLASSIFICATION: accuracy,
                                 REGRESSION: r2}
                evaluator = TrainEvaluator(backend_mock, queue_,
                                           resampling_strategy='cv',
                                           resampling_strategy_args={'folds': 2},
                                           output_y_hat_optimization=False,
                                           metric=metric_lookup[D.info['task']])

                evaluator.fit_predict_and_loss()
                rval = evaluator.queue.get(timeout=1)
                self.assertTrue(np.isfinite(rval['loss']))

    ############################################################################
    # Test obtaining a splitter object from scikit-learn
    @unittest.mock.patch.object(TrainEvaluator, "__init__")
    def test_get_splitter(self, te_mock):
        te_mock.return_value = None
        D = unittest.mock.Mock(spec=AbstractDataManager)
        D.data = dict(Y_train=np.array([0, 0, 0, 1, 1, 1]))
        D.info = dict(task=BINARY_CLASSIFICATION)
        D.feat_type = []

        # holdout, binary classification
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection.StratifiedShuffleSplit)

        # holdout, binary classification, fallback to shuffle split
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1, 2])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.ShuffleSplit)

        # cv, binary classification
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.StratifiedKFold)

        # cv, binary classification, no fallback anticipated
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1, 2])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.StratifiedKFold)

        # regression, shuffle split
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.ShuffleSplit)

        # regression cv, KFold
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.KFold)


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.queue = multiprocessing.Queue()
        self.configuration = get_configuration_space(
            {'task': MULTICLASS_CLASSIFICATION,
             'is_sparse': False}).get_default_configuration()
        self.data = get_multiclass_classification_datamanager()
        self.tmp_dir = os.path.join(os.path.dirname(__file__),
                                    '.test_holdout_functions')
        self.n = len(self.data.data['Y_train'])
        self.y = self.data.data['Y_train'].flatten()
        self.backend = unittest.mock.Mock()
        self.backend.get_model_dir.return_value = 'udiaetzrpduaeirdaetr'
        self.backend.load_datamanager.return_value = self.data
        self.backend.output_directory = 'duapdbaetpdbe'
        self.dataset_name = json.dumps({'task_id': 'test'})

    def test_eval_holdout(self):
        eval_holdout(
            queue=self.queue,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            all_scoring_functions=False,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        info = get_last_result(self.queue)
        self.assertAlmostEqual(info['loss'], 0.030303030303030276, places=3)
        self.assertEqual(info['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', info['additional_run_info'])

    def test_eval_holdout_all_loss_functions(self):
        eval_holdout(
            queue=self.queue,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            all_scoring_functions=True,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = get_last_result(self.queue)

        fixture = {'accuracy': 0.030303030303030276,
                   'balanced_accuracy': 0.033333333333333326,
                   'f1_macro': 0.032036613272311221,
                   'f1_micro': 0.030303030303030276,
                   'f1_weighted': 0.030441716940572849,
                   'log_loss': 1.0634089940876672,
                   'pac_score': 0.092288218582651682,
                   'precision_macro': 0.02777777777777779,
                   'precision_micro': 0.030303030303030276,
                   'precision_weighted': 0.027777777777777901,
                   'recall_macro': 0.033333333333333326,
                   'recall_micro': 0.030303030303030276,
                   'recall_weighted': 0.030303030303030276,
                   'num_run': 1}

        additional_run_info = rval['additional_run_info']
        for key, value in fixture.items():
            self.assertAlmostEqual(additional_run_info[key], fixture[key],
                                   msg=key)
        self.assertIn('duration', additional_run_info)
        self.assertEqual(len(additional_run_info), len(fixture) + 1,
                         msg=sorted(additional_run_info.items()))

        self.assertAlmostEqual(rval['loss'], 0.030303030303030276, places=3)
        self.assertEqual(rval['status'], StatusType.SUCCESS)

    # def test_eval_holdout_on_subset(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_holdout(self.queue, self.configuration, self.data,
    #                  backend_api, 1, 1, 43, True, False, True, None, None,
    #                  False)
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.1)
    #     self.assertEqual(info[2], 1)

    def test_eval_holdout_iterative_fit_no_timeout(self):
        eval_iterative_holdout(
            queue=self.queue,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            all_scoring_functions=False,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = get_last_result(self.queue)
        self.assertAlmostEqual(rval['loss'], 0.030303030303030276)
        self.assertEqual(rval['status'], StatusType.SUCCESS)

    # def test_eval_holdout_iterative_fit_on_subset_no_timeout(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_iterative_holdout(self.queue, self.configuration,
    #                            self.data, backend_api, 1, 1, 43, True, False,
    #                            True, None, None, False)
    #
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.1)
    #     self.assertEqual(info[2], 1)

    def test_eval_cv(self):
        eval_cv(
            queue=self.queue,
            config=self.configuration,
            backend=self.backend,
            seed=1,
            num_run=1,
            resampling_strategy='cv',
            resampling_strategy_args={'folds': 3},
            all_scoring_functions=False,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = get_last_result(self.queue)
        self.assertAlmostEqual(rval['loss'], 0.06)
        self.assertEqual(rval['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', rval['additional_run_info'])

    def test_eval_cv_all_loss_functions(self):
        eval_cv(
            queue=self.queue,
            config=self.configuration,
            backend=self.backend,
            seed=1,
            num_run=1,
            resampling_strategy='cv',
            resampling_strategy_args={'folds': 3},
            all_scoring_functions=True,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = get_last_result(self.queue)

        fixture = {'accuracy': 0.06,
                   'balanced_accuracy': 0.063508064516129004,
                   'f1_macro': 0.063508064516129004,
                   'f1_micro': 0.06,
                   'f1_weighted': 0.06,
                   'log_loss': 1.1299444831535221,
                   'pac_score': 0.18306366567302557,
                   'precision_macro': 0.063508064516129004,
                   'precision_micro': 0.06,
                   'precision_weighted': 0.06,
                   'recall_macro': 0.063508064516129004,
                   'recall_micro': 0.06,
                   'recall_weighted': 0.06,
                   'num_run': 1}

        additional_run_info = rval['additional_run_info']
        for key, value in fixture.items():
            self.assertAlmostEqual(additional_run_info[key], fixture[key], msg=key)
        self.assertIn('duration', additional_run_info)
        self.assertEqual(len(additional_run_info), len(fixture) + 1,
                         msg=sorted(additional_run_info.items()))

        self.assertAlmostEqual(rval['loss'], 0.06)
        self.assertEqual(rval['status'], StatusType.SUCCESS)

    # def test_eval_cv_on_subset(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_cv(queue=self.queue, config=self.configuration, data=self.data,
    #             backend=backend_api, seed=1, num_run=1, folds=5, subsample=45,
    #             with_predictions=True, all_scoring_functions=False,
    #             output_y_hat_optimization=True, include=None, exclude=None,
    #             disable_file_output=False)
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.063004032258064502)
    #     self.assertEqual(info[2], 1)

    def test_eval_partial_cv(self):
        results = [0.090909090909090939,
                   0.047619047619047672,
                   0.052631578947368474,
                   0.15789473684210531,
                   0.0]
        for fold in range(5):
            instance = json.dumps({'task_id': 'data', 'fold': fold})
            eval_partial_cv(
                queue=self.queue,
                config=self.configuration,
                backend=self.backend,
                seed=1,
                num_run=1,
                instance=instance,
                resampling_strategy='partial-cv',
                resampling_strategy_args={'folds': 5},
                all_scoring_functions=False,
                output_y_hat_optimization=True,
                include=None,
                exclude=None,
                disable_file_output=False,
                metric=accuracy,
            )
            rval = get_last_result(self.queue)
            self.assertAlmostEqual(rval['loss'], results[fold])
            self.assertEqual(rval['status'], StatusType.SUCCESS)

    # def test_eval_partial_cv_on_subset_no_timeout(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #
    #     results = [0.071428571428571508,
    #                0.15476190476190488,
    #                0.08333333333333337,
    #                0.16666666666666674,
    #                0.0]
    #     for fold in range(5):
    #         eval_partial_cv(queue=self.queue, config=self.configuration,
    #                         data=self.data, backend=backend_api,
    #                         seed=1, num_run=1, instance=fold, folds=5,
    #                         subsample=80, with_predictions=True,
    #                         all_scoring_functions=False, output_y_hat_optimization=True,
    #                         include=None, exclude=None,
    #                         disable_file_output=False)
    #
    #         info = self.queue.get()
    #         self.assertAlmostEqual(info[1], results[fold])
    #         self.assertEqual(info[2], 1)
    #
    #     results = [0.071428571428571508,
    #                0.15476190476190488,
    #                0.16666666666666674,
    #                0.0,
    #                0.0]
    #     for fold in range(5):
    #         eval_partial_cv(queue=self.queue, config=self.configuration,
    #                         data=self.data, backend=backend_api,
    #                         seed=1, num_run=1, instance=fold, folds=5,
    #                         subsample=43, with_predictions=True,
    #                         all_scoring_functions=False, output_y_hat_optimization=True,
    #                         include=None, exclude=None,
    #                         disable_file_output=False)
    #
    #         info = self.queue.get()
    #         self.assertAlmostEqual(info[1], results[fold])
    #         self.assertEqual(info[2], 1)

