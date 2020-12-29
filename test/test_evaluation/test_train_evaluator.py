import copy
import json
import logging.handlers
import queue
import multiprocessing
import os
import tempfile
import shutil
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, \
    KFold, LeaveOneGroupOut, LeavePGroupsOut, LeaveOneOut, LeavePOut, \
    PredefinedSplit, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, \
    StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit
import sklearn.model_selection
from smac.tae import StatusType, TAEAbortException

from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.evaluation.util import read_queue
from autosklearn.evaluation.train_evaluator import TrainEvaluator, \
    eval_holdout, eval_iterative_holdout, eval_cv, eval_partial_cv, subsample_indices
from autosklearn.util import backend
from autosklearn.util.pipeline import get_configuration_space
from autosklearn.constants import BINARY_CLASSIFICATION, \
    MULTILABEL_CLASSIFICATION,\
    MULTICLASS_CLASSIFICATION,\
    REGRESSION,\
    MULTIOUTPUT_REGRESSION
from autosklearn.metrics import accuracy, r2, f1_macro

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters, \
    get_multiclass_classification_datamanager, SCORER_LIST  # noqa (E402: module level import not at top of file)


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class TestTrainEvaluator(BaseEvaluatorTest, unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        """
        Creates a backend mock
        """
        tmp_dir_name = self.id()
        self.ev_path = os.path.join(this_directory, '.tmp_evaluations', tmp_dir_name)
        if os.path.exists(self.ev_path):
            shutil.rmtree(self.ev_path)
        os.makedirs(self.ev_path, exist_ok=False)
        dummy_model_files = [os.path.join(self.ev_path, str(n)) for n in range(100)]
        dummy_pred_files = [os.path.join(self.ev_path, str(n)) for n in range(100, 200)]
        dummy_cv_model_files = [os.path.join(self.ev_path, str(n)) for n in range(200, 300)]
        backend_mock = unittest.mock.Mock()
        backend_mock.temporary_directory = tempfile.gettempdir()
        backend_mock.get_model_dir.return_value = self.ev_path
        backend_mock.get_cv_model_dir.return_value = self.ev_path
        backend_mock.get_model_path.side_effect = dummy_model_files
        backend_mock.get_cv_model_path.side_effect = dummy_cv_model_files
        backend_mock.get_prediction_output_path.side_effect = dummy_pred_files
        self.backend_mock = backend_mock

        self.tmp_dir = os.path.join(self.ev_path, 'tmp_dir')
        self.output_dir = os.path.join(self.ev_path, 'out_dir')

        self.port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

    def tearDown(self):
        if os.path.exists(self.ev_path):
            shutil.rmtree(self.ev_path)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_holdout(self, pipeline_mock):
        # Binary iris, contains 69 train samples, 25 validation samples,
        # 6 test samples
        D = get_binary_classification_datamanager()
        D.name = 'test'

        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.get_max_iter.return_value = 1
        pipeline_mock.get_current_iter.return_value = 1

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   configuration=configuration,
                                   resampling_strategy='holdout',
                                   resampling_strategy_args={'train_size': 0.66},
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy,
                                   port=self.port,
                                   )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 0.45833333333333337)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # four calls because of train, holdout, validation and test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 4)
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 24)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0],
                         D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.data['Y_test'].shape[0])
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
                return self.fully_fitted_call_count > 18

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = \
            SideEffect().configuration_fully_fitted
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_max_iter.return_value = 512
        pipeline_mock.get_current_iter.side_effect = (2, 4, 8, 16, 32, 64, 128, 256, 512)

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='holdout',
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy,
                                   budget=0.0)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 1.0, 1.0, 1.0,
                               0.9, 0.9, 0.9, 0.9,
                               0.8, 0.8, 0.8, 0.8,
                               0.7, 0.7, 0.7, 0.7,
                               0.6, 0.6, 0.6, 0.6,
                               0.5, 0.5, 0.5, 0.5,
                               0.4, 0.4, 0.4, 0.4,
                               0.3, 0.3, 0.3, 0.3,
                               0.2, 0.2, 0.2, 0.2]
                self.iteration = 0

            def side_effect(self, *args, **kwargs):
                self.iteration += 1
                return self.losses[self.iteration - 1]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 9)

        for i in range(1, 10):
            rval = evaluator.queue.get(timeout=1)
            result = rval['loss']
            self.assertAlmostEqual(result, 1.0 - (0.1 * (i - 1)))
            if i < 9:
                self.assertEqual(rval['status'], StatusType.DONOTADVANCE)
                self.assertEqual(len(rval), 3)
            else:
                self.assertEqual(rval['status'], StatusType.SUCCESS)
                self.assertEqual(len(rval), 4)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 9)
        self.assertEqual(
            [cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list],
            [2, 2, 4, 8, 16, 32, 64, 128, 256]
        )
        # 20 calls because of train, holdout, validation and test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 36)
        # 1/3 of 69
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 23)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0],
                         D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 9)
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
                    raise ValueError('fixture')
                return self.fully_fitted_call_count > 10

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = \
            SideEffect().configuration_fully_fitted
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.get_max_iter.return_value = 512
        pipeline_mock.get_current_iter.side_effect = (2, 4, 8, 16, 32, 64, 128, 256, 512)

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='holdout-iterative-fit',
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy,
                                   budget=0.0)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [0.8, 0.8, 0.8, 0.8,
                               0.6, 0.6, 0.6, 0.6,
                               0.4, 0.4, 0.4, 0.4,
                               0.2, 0.2, 0.2, 0.2,
                               0.0, 0.0, 0.0, 0.0]
                self.iteration = 0

            def side_effect(self, *args, **kwargs):
                self.iteration += 1
                return self.losses[self.iteration - 1]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        self.assertRaisesRegex(
            ValueError,
            'fixture',
            evaluator.fit_predict_and_loss,
            iterative=True,
        )
        self.assertEqual(evaluator.file_output.call_count, 2)

        for i in range(1, 3):
            rval = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(rval['loss'], 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 2)
        # eight calls because of train, holdout, the validation and the test set
        # and a total of two calls each because of two iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 8)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 23)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0],
                         D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.data['Y_test'].shape[0])
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
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='holdout-iterative-fit',
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 1)

        rval = evaluator.queue.get(timeout=1)
        self.assertAlmostEqual(rval['loss'], 0.47826086956521741)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 0)
        # four calls for train, opt, valid and test
        self.assertEqual(evaluator.model.predict_proba.call_count, 4)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 23)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0],
                         D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()

        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 5},
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.fit_predict_and_loss()

        rval = read_queue(evaluator.queue)
        self.assertEqual(len(rval), 1)
        result = rval[0]['loss']
        self.assertEqual(len(rval[0]), 3)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 0.463768115942029)
        self.assertEqual(pipeline_mock.fit.call_count, 5)
        # Fifteen calls because of the training, holdout, validation and
        # test set (4 sets x 5 folds = 20)
        self.assertEqual(pipeline_mock.predict_proba.call_count, 20)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0],
                         D.data['Y_train'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0],
                         D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0],
                         D.data['Y_test'].shape[0])
        # The model prior to fitting is saved, this cannot be directly tested
        # because of the way the mock module is used. Instead, we test whether
        # the if block in which model assignment is done is accessed
        self.assertTrue(evaluator._added_empty_model)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_partial_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()

        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.get_max_iter.return_value = 1
        pipeline_mock.get_current_iter.return_value = 1
        D = get_binary_classification_datamanager()
        D.name = 'test'

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='partial-cv',
                                   resampling_strategy_args={'folds': 5},
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)

        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.partial_fit_predict_and_loss(fold=1)

        rval = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 0)
        self.assertEqual(rval['loss'], 0.5)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        self.assertEqual(pipeline_mock.predict_proba.call_count, 4)
        # The model prior to fitting is saved, this cannot be directly tested
        # because of the way the mock module is used. Instead, we test whether
        # the if block in which model assignment is done is accessed
        self.assertTrue(hasattr(evaluator, 'model'))

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
                return self.fully_fitted_call_count > 18

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = \
            SideEffect().configuration_fully_fitted
        pipeline_mock.fit_transformer.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = \
            lambda X, batch_size=None: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.get_additional_run_info.return_value = None
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        pipeline_mock.get_max_iter.return_value = 512
        pipeline_mock.get_current_iter.side_effect = (2, 4, 8, 16, 32, 64, 128, 256, 512)

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(self.tmp_dir, self.output_dir)
        backend_api.load_datamanager = lambda: D
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(backend_api, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='partial-cv-iterative-fit',
                                   resampling_strategy_args={'folds': 5},
                                   scoring_functions=None,
                                   output_y_hat_optimization=True,
                                   metric=accuracy,
                                   budget=0.0)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 1.0, 1.0, 1.0,
                               0.9, 0.9, 0.9, 0.9,
                               0.8, 0.8, 0.8, 0.8,
                               0.7, 0.7, 0.7, 0.7,
                               0.6, 0.6, 0.6, 0.6,
                               0.5, 0.5, 0.5, 0.5,
                               0.4, 0.4, 0.4, 0.4,
                               0.3, 0.3, 0.3, 0.3,
                               0.2, 0.2, 0.2, 0.2]
                self.iteration = 0

            def side_effect(self, *args, **kwargs):
                self.iteration += 1
                return self.losses[self.iteration - 1]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        evaluator.partial_fit_predict_and_loss(fold=1, iterative=True)
        # No file output here!
        self.assertEqual(evaluator.file_output.call_count, 0)

        for i in range(1, 10):
            rval = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(rval['loss'], 1.0 - (0.1 * (i - 1)))
            if i < 9:
                self.assertEqual(rval['status'], StatusType.DONOTADVANCE)
            else:
                self.assertEqual(rval['status'], StatusType.SUCCESS)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 9)
        self.assertEqual(
            [cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list],
            [2, 2, 4, 8, 16, 32, 64, 128, 256]
        )
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertTrue(hasattr(evaluator, 'model'))
        self.assertEqual(pipeline_mock.iterative_fit.call_count, 9)
        # 20 calls because of train, holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(pipeline_mock.predict_proba.call_count, 36)

    @unittest.mock.patch.object(TrainEvaluator, '_loss')
    def test_file_output(self, loss_mock):

        D = get_regression_datamanager()
        D.name = 'test'
        self.backend_mock.load_datamanager.return_value = D
        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()
        loss_mock.return_value = None

        evaluator = TrainEvaluator(self.backend_mock, queue=queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 5},
                                   scoring_functions=SCORER_LIST,
                                   output_y_hat_optimization=True,
                                   metric=accuracy)

        self.backend_mock.get_model_dir.return_value = True
        evaluator.model = 'model'
        evaluator.Y_optimization = D.data['Y_train']
        rval = evaluator.file_output(
            D.data['Y_train'],
            D.data['Y_valid'],
            D.data['Y_test'],
        )

        self.assertEqual(rval, (None, {}))
        self.assertEqual(self.backend_mock.save_targets_ensemble.call_count, 1)
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, 1)
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1].keys(),
                         {'seed', 'idx', 'budget', 'model', 'cv_model',
                          'ensemble_predictions', 'valid_predictions', 'test_predictions'})
        self.assertIsNotNone(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
        self.assertIsNone(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])

        evaluator.models = ['model2', 'model2']
        rval = evaluator.file_output(
            D.data['Y_train'],
            D.data['Y_valid'],
            D.data['Y_test'],
        )
        self.assertEqual(rval, (None, {}))
        self.assertEqual(self.backend_mock.save_targets_ensemble.call_count, 2)
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, 2)
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1].keys(),
                         {'seed', 'idx', 'budget', 'model', 'cv_model',
                          'ensemble_predictions', 'valid_predictions', 'test_predictions'})
        self.assertIsNotNone(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
        self.assertIsNotNone(self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])

        # Check for not containing NaNs - that the models don't predict nonsense
        # for unseen data
        D.data['Y_valid'][0] = np.NaN
        rval = evaluator.file_output(
            D.data['Y_train'],
            D.data['Y_valid'],
            D.data['Y_test'],
        )
        self.assertEqual(
            rval,
            (
                1.0,
                {
                    'error':
                    'Model predictions for validation set contains NaNs.'
                },
            )
        )
        D.data['Y_train'][0] = np.NaN
        rval = evaluator.file_output(
            D.data['Y_train'],
            D.data['Y_valid'],
            D.data['Y_test'],
        )
        self.assertEqual(
            rval,
            (
                1.0,
                {
                    'error':
                    'Model predictions for optimization set contains NaNs.'
                 },
            )
        )

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_subsample_indices_classification(self, mock, backend_mock):

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()
        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        evaluator = TrainEvaluator(backend_mock, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 10},
                                   metric=accuracy)
        train_indices = np.arange(69, dtype=int)
        train_indices1 = subsample_indices(
            train_indices, 0.1449, evaluator.task_type, evaluator.Y_train)
        evaluator.subsample = 20
        train_indices2 = subsample_indices(
            train_indices, 0.2898, evaluator.task_type, evaluator.Y_train)
        evaluator.subsample = 30
        train_indices3 = subsample_indices(
            train_indices, 0.4347, evaluator.task_type, evaluator.Y_train)
        evaluator.subsample = 67
        train_indices4 = subsample_indices(
            train_indices, 0.971, evaluator.task_type, evaluator.Y_train)
        # Common cases
        for ti in train_indices1:
            self.assertIn(ti, train_indices2)
        for ti in train_indices2:
            self.assertIn(ti, train_indices3)
        for ti in train_indices3:
            self.assertIn(ti, train_indices4)

        # Corner cases
        self.assertRaisesRegex(
            ValueError, 'train_size=0.0 should be either positive and smaller than the '
            r'number of samples 69 or a float in the \(0, 1\) range',
            subsample_indices, train_indices, 0.0, evaluator.task_type, evaluator.Y_train)
        # With equal or greater it should return a non-shuffled array of indices
        train_indices5 = subsample_indices(
            train_indices, 1.0, evaluator.task_type, evaluator.Y_train)
        self.assertTrue(np.all(train_indices5 == train_indices))
        evaluator.subsample = 68
        self.assertRaisesRegex(
            ValueError, 'The test_size = 1 should be greater or equal to the number of '
            'classes = 2', subsample_indices, train_indices, 0.9999, evaluator.task_type,
            evaluator.Y_train)

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_subsample_indices_regression(self, mock, backend_mock):

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()
        backend_mock.temporary_directory = tempfile.gettempdir()
        evaluator = TrainEvaluator(backend_mock, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 10},
                                   metric=accuracy)
        train_indices = np.arange(69, dtype=int)
        train_indices3 = subsample_indices(train_indices, subsample=0.4347,
                                           task_type=evaluator.task_type,
                                           Y_train=evaluator.Y_train)
        evaluator.subsample = 67
        train_indices4 = subsample_indices(train_indices, subsample=0.4347,
                                           task_type=evaluator.task_type,
                                           Y_train=evaluator.Y_train)
        # Common cases
        for ti in train_indices3:
            self.assertIn(ti, train_indices4)

        # Corner cases
        self.assertRaisesRegex(
            ValueError, 'train_size=0.0 should be either positive and smaller than the '
            r'number of samples 69 or a float in the \(0, 1\) range',
            subsample_indices, train_indices, 0.0,
            evaluator.task_type, evaluator.Y_train)
        self.assertRaisesRegex(
            ValueError, 'Subsample must not be larger than 1, but is 1.000100',
            subsample_indices, train_indices, 1.0001,
            evaluator.task_type, evaluator.Y_train)
        # With equal or greater it should return a non-shuffled array of indices
        train_indices6 = subsample_indices(train_indices, 1.0, evaluator.task_type,
                                           evaluator.Y_train)
        np.testing.assert_allclose(train_indices6, train_indices)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_predict_proba_binary_classification(self, mock):
        D = get_binary_classification_datamanager()
        self.backend_mock.load_datamanager.return_value = D
        mock.predict_proba.side_effect = lambda y, batch_size=None: np.array(
            [[0.1, 0.9]] * y.shape[0]
        )
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(self.backend_mock, queue_,
                                   port=self.port,
                                   configuration=configuration,
                                   resampling_strategy='cv',
                                   resampling_strategy_args={'folds': 10},
                                   output_y_hat_optimization=False,
                                   metric=accuracy)

        evaluator.fit_predict_and_loss()
        Y_optimization_pred = self.backend_mock.save_numrun_to_dir.call_args_list[0][1][
            'ensemble_predictions']

        for i in range(7):
            self.assertEqual(0.9, Y_optimization_pred[i][1])

    @unittest.mock.patch.object(TrainEvaluator, 'file_output')
    @unittest.mock.patch.object(TrainEvaluator, '_partial_fit_and_predict_standard')
    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_fit_predict_and_loss_standard_additional_run_info(
        self, mock, backend_mock, _partial_fit_and_predict_mock,
            file_output_mock,
    ):
        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        mock.side_effect = lambda **kwargs: mock
        _partial_fit_and_predict_mock.return_value = (
            np.array([[0.1, 0.9]] * 46),
            np.array([[0.1, 0.9]] * 23),
            np.array([[0.1, 0.9]] * 25),
            np.array([[0.1, 0.9]] * 6),
            {'a': 5},
        )
        file_output_mock.return_value = (None, {})

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(
            backend_mock, queue_,
            port=self.port,
            configuration=configuration,
            resampling_strategy='holdout',
            output_y_hat_optimization=False,
            metric=accuracy,
        )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})
        evaluator.model = unittest.mock.Mock()
        evaluator.model.estimator_supports_iterative_fit.return_value = False
        evaluator.Y_targets[0] = np.array([1] * 23)
        evaluator.Y_train_targets = np.array([1] * 69)
        rval = evaluator.fit_predict_and_loss(iterative=False)
        self.assertIsNone(rval)
        element = queue_.get()
        self.assertEqual(element['status'], StatusType.SUCCESS)
        self.assertEqual(element['additional_run_info']['a'], 5)
        self.assertEqual(_partial_fit_and_predict_mock.call_count, 1)

        class SideEffect(object):
            def __init__(self):
                self.n_call = 0

            def __call__(self, *args, **kwargs):
                if self.n_call == 0:
                    self.n_call += 1
                    return (
                        np.array([[0.1, 0.9]] * 34),
                        np.array([[0.1, 0.9]] * 35),
                        np.array([[0.1, 0.9]] * 25),
                        np.array([[0.1, 0.9]] * 6),
                        {'a': 5}
                    )
                else:
                    return (
                        np.array([[0.1, 0.9]] * 34),
                        np.array([[0.1, 0.9]] * 34),
                        np.array([[0.1, 0.9]] * 25),
                        np.array([[0.1, 0.9]] * 6),
                        {'a': 5}
                    )
        _partial_fit_and_predict_mock.side_effect = SideEffect()
        evaluator = TrainEvaluator(
            backend_mock, queue_,
            port=self.port,
            configuration=configuration,
            resampling_strategy='cv',
            resampling_strategy_args={'folds': 2},
            output_y_hat_optimization=False,
            metric=accuracy,
        )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})
        evaluator.Y_targets[0] = np.array([1] * 35)
        evaluator.Y_targets[1] = np.array([1] * 34)
        evaluator.Y_train_targets = np.array([1] * 69)

        self.assertRaisesRegex(
            TAEAbortException,
            'Found additional run info "{\'a\': 5}" in fold 1, '
            'but cannot handle additional run info if fold >= 1.',
            evaluator.fit_predict_and_loss,
            iterative=False
        )

    @unittest.mock.patch.object(TrainEvaluator, '_loss')
    @unittest.mock.patch.object(TrainEvaluator, 'finish_up')
    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_fit_predict_and_loss_iterative_additional_run_info(
            self, mock, backend_mock, finish_up_mock, loss_mock,
    ):

        class Counter:
            counter = 0

            def __call__(self):
                self.counter += 1
                return False if self.counter <= 1 else True
        mock.estimator_supports_iterative_fit.return_value = True
        mock.fit_transformer.return_value = ('Xt', {})
        mock.configuration_fully_fitted.side_effect = Counter()
        mock.get_current_iter.side_effect = Counter()
        mock.get_max_iter.return_value = 1
        mock.get_additional_run_info.return_value = 14678
        loss_mock.return_value = 0.5

        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(
            backend_mock, queue_,
            port=self.port,
            configuration=configuration,
            resampling_strategy='holdout',
            output_y_hat_optimization=False,
            metric=accuracy,
            budget=0.0
        )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})
        evaluator.Y_targets[0] = np.array([1] * 23).reshape((-1, 1))
        evaluator.Y_train_targets = np.array([1] * 69).reshape((-1, 1))
        rval = evaluator.fit_predict_and_loss(iterative=True)
        self.assertIsNone(rval)
        self.assertEqual(finish_up_mock.call_count, 1)
        self.assertEqual(finish_up_mock.call_args[1]['additional_run_info'], 14678)

    @unittest.mock.patch.object(TrainEvaluator, '_loss')
    @unittest.mock.patch.object(TrainEvaluator, 'finish_up')
    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_fit_predict_and_loss_iterative_noniterativemodel_additional_run_info(
            self, mock, backend_mock, finish_up_mock, loss_mock,
    ):
        mock.estimator_supports_iterative_fit.return_value = False
        mock.fit_transformer.return_value = ('Xt', {})
        mock.get_additional_run_info.return_value = 14678
        loss_mock.return_value = 0.5

        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(
            backend_mock, queue_,
            port=self.port,
            configuration=configuration,
            resampling_strategy='holdout',
            output_y_hat_optimization=False,
            metric=accuracy,
        )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.Y_targets[0] = np.array([1] * 23).reshape((-1, 1))
        evaluator.Y_train_targets = np.array([1] * 69).reshape((-1, 1))
        rval = evaluator.fit_predict_and_loss(iterative=True)
        self.assertIsNone(rval)
        self.assertEqual(finish_up_mock.call_count, 1)
        self.assertEqual(finish_up_mock.call_args[1]['additional_run_info'], 14678)

    @unittest.mock.patch.object(TrainEvaluator, '_loss')
    @unittest.mock.patch.object(TrainEvaluator, 'finish_up')
    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_fit_predict_and_loss_budget_additional_run_info(
            self, mock, backend_mock, finish_up_mock, loss_mock,
    ):
        class Counter:
            counter = 0

            def __call__(self):
                self.counter += 1
                return False if self.counter <= 1 else True
        mock.configuration_fully_fitted.side_effect = Counter()
        mock.get_current_iter.side_effect = Counter()
        mock.get_max_iter.return_value = 1
        mock.estimator_supports_iterative_fit.return_value = True
        mock.fit_transformer.return_value = ('Xt', {})
        mock.get_additional_run_info.return_value = {'val': 14678}
        mock.get_max_iter.return_value = 512
        loss_mock.return_value = 0.5

        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(
            backend_mock, queue_,
            port=self.port,
            configuration=configuration,
            resampling_strategy='holdout',
            output_y_hat_optimization=False,
            metric=accuracy,
            budget_type='iterations',
            budget=50,
        )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.Y_targets[0] = np.array([1] * 23).reshape((-1, 1))
        evaluator.Y_train_targets = np.array([1] * 69).reshape((-1, 1))
        rval = evaluator.fit_predict_and_loss(iterative=False)
        self.assertIsNone(rval)
        self.assertEqual(finish_up_mock.call_count, 1)
        self.assertEqual(finish_up_mock.call_args[1]['additional_run_info'], {'val': 14678})

    @unittest.mock.patch.object(TrainEvaluator, '_loss')
    @unittest.mock.patch.object(TrainEvaluator, 'finish_up')
    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_fit_predict_and_loss_budget_2_additional_run_info(
            self, mock, backend_mock, finish_up_mock, loss_mock,
    ):
        mock.estimator_supports_iterative_fit.return_value = False
        mock.fit_transformer.return_value = ('Xt', {})
        mock.get_additional_run_info.return_value = {'val': 14678}
        loss_mock.return_value = 0.5

        D = get_binary_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(
            backend_mock, queue_,
            port=self.port,
            configuration=configuration,
            resampling_strategy='holdout',
            output_y_hat_optimization=False,
            metric=accuracy,
            budget_type='subsample',
            budget=50,
        )
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, {})

        evaluator.Y_targets[0] = np.array([1] * 23).reshape((-1, 1))
        evaluator.Y_train_targets = np.array([1] * 69).reshape((-1, 1))
        rval = evaluator.fit_predict_and_loss(iterative=False)
        self.assertIsNone(rval)
        self.assertEqual(finish_up_mock.call_count, 1)
        self.assertEqual(finish_up_mock.call_args[1]['additional_run_info'], {'val': 14678})

    def test_get_results(self):
        queue_ = multiprocessing.Queue()
        for i in range(5):
            queue_.put((i * 1, 1 - (i * 0.2), 0, "", StatusType.SUCCESS))
        result = read_queue(queue_)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(result[0][1], 1.0)

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)

            with self.subTest(testname):
                D = getter()
                D_ = copy.deepcopy(D)
                y = D.data['Y_train']
                if len(y.shape) == 2 and y.shape[1] == 1:
                    D_.data['Y_train'] = y.flatten()
                self.backend_mock.load_datamanager.return_value = D_
                queue_ = multiprocessing.Queue()
                metric_lookup = {MULTILABEL_CLASSIFICATION: f1_macro,
                                 BINARY_CLASSIFICATION: accuracy,
                                 MULTICLASS_CLASSIFICATION: accuracy,
                                 REGRESSION: r2}
                evaluator = TrainEvaluator(self.backend_mock, queue_,
                                           port=self.port,
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
        evaluator.resampling_strategy_args = {}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection.StratifiedShuffleSplit)

        # holdout, binary classification, no shuffle
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = {'shuffle': False}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection.PredefinedSplit)

        # holdout, binary classification, fallback to shuffle split
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1, 2])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = {}
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

        # cv, binary classification, shuffle is True
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.StratifiedKFold)
        self.assertTrue(cv.shuffle)

        # cv, binary classification, shuffle is False
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5, 'shuffle': False}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.KFold)
        self.assertFalse(cv.shuffle)

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
        evaluator.resampling_strategy_args = {}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.ShuffleSplit)

        # regression, no shuffle
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = {'shuffle': False}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.PredefinedSplit)

        # regression cv, KFold
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.KFold)
        self.assertTrue(cv.shuffle)

        # regression cv, KFold, no shuffling
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5, 'shuffle': False}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.KFold)
        self.assertFalse(cv.shuffle)

        # multioutput regression, shuffle split
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = {}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.ShuffleSplit)

        # multioutput regression, no shuffle
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'
        evaluator.resampling_strategy_args = {'shuffle': False}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv,
                              sklearn.model_selection._split.PredefinedSplit)

        # multioutput regression cv, KFold
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.KFold)
        self.assertTrue(cv.shuffle)

        # multioutput regression cv, KFold, no shuffling
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'cv'
        evaluator.resampling_strategy_args = {'folds': 5, 'shuffle': False}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, sklearn.model_selection._split.KFold)
        self.assertFalse(cv.shuffle)

    @unittest.mock.patch.object(TrainEvaluator, "__init__")
    def test_get_splitter_cv_object(self, te_mock):
        te_mock.return_value = None
        D = unittest.mock.Mock(spec=AbstractDataManager)
        D.data = dict(Y_train=np.array([0, 0, 0, 1, 1, 1]))
        D.info = dict(task=BINARY_CLASSIFICATION)
        D.feat_type = []

        # GroupKFold, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupKFold
        evaluator.resampling_strategy_args = {'folds': 2, 'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, GroupKFold)
        self.assertEqual(cv.get_n_splits(groups=evaluator.resampling_strategy_args['groups']), 2)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupKFold, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupKFold
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # GroupKFold, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupKFold
        evaluator.resampling_strategy_args = {'folds': 2, 'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, GroupKFold)
        self.assertEqual(cv.get_n_splits(groups=evaluator.resampling_strategy_args['groups']), 2)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupKFold, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupKFold
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # GroupKFold, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupKFold
        evaluator.resampling_strategy_args = {'folds': 2, 'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, GroupKFold)
        self.assertEqual(cv.get_n_splits(groups=evaluator.resampling_strategy_args['groups']), 2)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupKFold, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupKFold
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # KFold, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = KFold
        evaluator.resampling_strategy_args = {'folds': 4, 'shuffle': True,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # KFold, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = KFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertFalse(cv.shuffle)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # KFold, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = KFold
        evaluator.resampling_strategy_args = {'folds': 4, 'shuffle': True,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # KFold, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = KFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertFalse(cv.shuffle)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # KFold, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = KFold
        evaluator.resampling_strategy_args = {'folds': 4, 'shuffle': True,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # KFold, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = KFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertFalse(cv.shuffle)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeaveOneGroupOut, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneGroupOut
        evaluator.resampling_strategy_args = {'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeaveOneGroupOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeaveOneGroupOut, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneGroupOut
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # LeaveOneGroupOut, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneGroupOut
        evaluator.resampling_strategy_args = {'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeaveOneGroupOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeaveOneGroupOut, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneGroupOut
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # LeaveOneGroupOut, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneGroupOut
        evaluator.resampling_strategy_args = {'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeaveOneGroupOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeaveOneGroupOut, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneGroupOut
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # LeavePGroupsOut, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePGroupsOut
        evaluator.resampling_strategy_args = {'n_groups': 1,
                                              'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePGroupsOut)
        self.assertEqual(cv.n_groups, 1)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePGroupsOut, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePGroupsOut
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # LeavePGroupsOut, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePGroupsOut
        evaluator.resampling_strategy_args = {'n_groups': 1,
                                              'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePGroupsOut)
        self.assertEqual(cv.n_groups, 1)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePGroupsOut, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePGroupsOut
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # LeavePGroupsOut, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePGroupsOut
        evaluator.resampling_strategy_args = {'n_groups': 1,
                                              'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePGroupsOut)
        self.assertEqual(cv.n_groups, 1)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePGroupsOut, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePGroupsOut
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # LeaveOneOut, classification
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneOut
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeaveOneOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeaveOneOut, regression
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneOut
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeaveOneOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeaveOneOut, multi-output regression
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeaveOneOut
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeaveOneOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePOut, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePOut
        evaluator.resampling_strategy_args = {'p': 3}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePOut)
        self.assertEqual(cv.p, 3)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePOut, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePOut
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePOut)
        self.assertEqual(cv.p, 2)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePOut, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePOut
        evaluator.resampling_strategy_args = {'p': 3}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePOut)
        self.assertEqual(cv.p, 3)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePOut, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePOut
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePOut)
        self.assertEqual(cv.p, 2)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePOut, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePOut
        evaluator.resampling_strategy_args = {'p': 3}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePOut)
        self.assertEqual(cv.p, 3)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # LeavePOut, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = LeavePOut
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, LeavePOut)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # PredefinedSplit, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = PredefinedSplit
        evaluator.resampling_strategy_args = {'test_fold': np.array([0, 1, 0, 1, 0, 1])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, PredefinedSplit)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # PredefinedSplit, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = PredefinedSplit
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter test_fold for class PredefinedSplit.',
            evaluator.get_splitter,
            D)

        # PredefinedSplit, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = PredefinedSplit
        evaluator.resampling_strategy_args = {'test_fold': np.array([0, 1, 0, 1, 0, 1])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, PredefinedSplit)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # PredefinedSplit, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = PredefinedSplit
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter test_fold for class PredefinedSplit.',
            evaluator.get_splitter,
            D)

        # PredefinedSplit, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = PredefinedSplit
        evaluator.resampling_strategy_args = {'test_fold': np.array([0, 1, 0, 1, 0, 1])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, PredefinedSplit)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # PredefinedSplit, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = PredefinedSplit
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter test_fold for class PredefinedSplit.',
            evaluator.get_splitter,
            D)

        # RepeatedKFold, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedKFold
        evaluator.resampling_strategy_args = {'folds': 4, 'n_repeats': 3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4*3)
        self.assertEqual(cv.n_repeats, 3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedKFold, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedKFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 5*10)
        self.assertEqual(cv.n_repeats, 10)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedKFold, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedKFold
        evaluator.resampling_strategy_args = {'folds': 4, 'n_repeats': 3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4*3)
        self.assertEqual(cv.n_repeats, 3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedKFold, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedKFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 5*10)
        self.assertEqual(cv.n_repeats, 10)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedKFold, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedKFold
        evaluator.resampling_strategy_args = {'folds': 4, 'n_repeats': 3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4*3)
        self.assertEqual(cv.n_repeats, 3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedKFold, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedKFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 5*10)
        self.assertEqual(cv.n_repeats, 10)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedStratifiedKFold, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedStratifiedKFold
        evaluator.resampling_strategy_args = {'folds': 2, 'n_repeats': 3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedStratifiedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2*3)
        self.assertEqual(cv.n_repeats, 3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # RepeatedStratifiedKFold, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = RepeatedStratifiedKFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, RepeatedStratifiedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 5*10)
        self.assertEqual(cv.n_repeats, 10)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # StratifiedKFold, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = StratifiedKFold
        evaluator.resampling_strategy_args = {'folds': 2, 'shuffle': True,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # StratifiedKFold, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = StratifiedKFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertFalse(cv.shuffle)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # TimeSeriesSplit, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = TimeSeriesSplit
        evaluator.resampling_strategy_args = {'folds': 4, 'max_train_size': 3}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, TimeSeriesSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4)
        self.assertEqual(cv.max_train_size, 3)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # TimeSeriesSplit, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = TimeSeriesSplit
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, TimeSeriesSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertIsNone(cv.max_train_size)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # TimeSeriesSplit, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = TimeSeriesSplit
        evaluator.resampling_strategy_args = {'folds': 4, 'max_train_size': 3}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, TimeSeriesSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 4)
        self.assertEqual(cv.max_train_size, 3)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # TimeSeriesSplit, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = TimeSeriesSplit
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, TimeSeriesSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertIsNone(cv.max_train_size)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # StratifiedKFold, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = StratifiedKFold
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 3)
        self.assertFalse(cv.shuffle)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupShuffleSplit, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5,
                                              'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, GroupShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupShuffleSplit, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupShuffleSplit
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # GroupShuffleSplit, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5,
                                              'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, GroupShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupShuffleSplit, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupShuffleSplit
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # GroupShuffleSplit, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5,
                                              'groups': np.array([1, 1, 2, 1, 2, 2])}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, GroupShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # GroupShuffleSplit, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = GroupShuffleSplit
        evaluator.resampling_strategy_args = None
        self.assertRaisesRegex(
            ValueError,
            'Must provide parameter groups for chosen CrossValidator.',
            evaluator.get_splitter,
            D)

        # StratifiedShuffleSplit, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = StratifiedShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, StratifiedShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # StratifiedShuffleSplit, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                      0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = StratifiedShuffleSplit
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, StratifiedShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 10)
        self.assertIsNone(cv.test_size)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # ShuffleSplit, classification with args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = ShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, ShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # ShuffleSplit, classification no args
        D.data['Y_train'] = np.array([0, 0, 0, 1, 1, 1])
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = ShuffleSplit
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, ShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 10)
        self.assertIsNone(cv.test_size)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # ShuffleSplit, regression with args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = ShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, ShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # ShuffleSplit, regression no args
        D.data['Y_train'] = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        D.info['task'] = REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = ShuffleSplit
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, ShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 10)
        self.assertIsNone(cv.test_size)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # ShuffleSplit, multi-output regression with args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = ShuffleSplit
        evaluator.resampling_strategy_args = {'folds': 2, 'test_size': 0.3,
                                              'random_state': 5}
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, ShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 2)
        self.assertEqual(cv.test_size, 0.3)
        self.assertEqual(cv.random_state, 5)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

        # ShuffleSplit, multi-output regression no args
        D.data['Y_train'] = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5],
                                     [1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
        D.info['task'] = MULTIOUTPUT_REGRESSION
        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = ShuffleSplit
        evaluator.resampling_strategy_args = None
        cv = evaluator.get_splitter(D)
        self.assertIsInstance(cv, ShuffleSplit)
        self.assertEqual(cv.get_n_splits(
            groups=evaluator.resampling_strategy_args['groups']), 10)
        self.assertIsNone(cv.test_size)
        self.assertIsNone(cv.random_state)
        next(cv.split(D.data['Y_train'], D.data['Y_train'],
                      groups=evaluator.resampling_strategy_args['groups']))

    @unittest.mock.patch.object(TrainEvaluator, "__init__")
    def test_holdout_split_size(self, te_mock):
        te_mock.return_value = None
        D = unittest.mock.Mock(spec=AbstractDataManager)
        D.feat_type = []

        evaluator = TrainEvaluator()
        evaluator.resampling_strategy = 'holdout'

        # Exact Ratio
        D.data = dict(Y_train=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
        D.info = dict(task=BINARY_CLASSIFICATION)
        evaluator.resampling_strategy_args = {'shuffle': True,
                                              'train_size': 0.7}
        cv = evaluator.get_splitter(D)

        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 7)
        self.assertEqual(len(test_samples), 3)

        # No Shuffle
        evaluator.resampling_strategy_args = {'shuffle': False,
                                              'train_size': 0.7}
        cv = evaluator.get_splitter(D)

        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 7)
        self.assertEqual(len(test_samples), 3)

        # Rounded Ratio
        D.data = dict(Y_train=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]))

        evaluator.resampling_strategy_args = {'shuffle': True,
                                              'train_size': 0.7}
        cv = evaluator.get_splitter(D)

        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 6)
        self.assertEqual(len(test_samples), 3)

        # Rounded Ratio No Shuffle
        evaluator.resampling_strategy_args = {'shuffle': False,
                                              'train_size': 0.7}
        cv = evaluator.get_splitter(D)

        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 6)
        self.assertEqual(len(test_samples), 3)

        # More data
        evaluator.resampling_strategy_args = {'shuffle': True,
                                              'train_size': 0.7}

        D.data = dict(Y_train=np.zeros((900, 1)))
        cv = evaluator.get_splitter(D)
        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 630)
        self.assertEqual(len(test_samples), 270)

        evaluator.resampling_strategy_args = {'train_size': 0.752}
        D.data = dict(Y_train=np.zeros((900, 1)))
        cv = evaluator.get_splitter(D)
        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 676)
        self.assertEqual(len(test_samples), 224)

        # Multilabel Exact Ratio
        D.data = dict(Y_train=np.array([[0, 0], [0, 1], [1, 1], [1, 0], [1, 1],
                                        [1, 1], [1, 1], [1, 0], [1, 1], [1, 1]]
                                       ))
        D.info = dict(task=MULTILABEL_CLASSIFICATION)
        evaluator.resampling_strategy_args = {'shuffle': True,
                                              'train_size': 0.7}
        cv = evaluator.get_splitter(D)

        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 7)
        self.assertEqual(len(test_samples), 3)

        # Multilabel No Shuffle
        D.data = dict(Y_train=np.array([[0, 0], [0, 1], [1, 1], [1, 0], [1, 1],
                                        [1, 1], [1, 1], [1, 0], [1, 1]]))
        evaluator.resampling_strategy_args = {'shuffle': False,
                                              'train_size': 0.7}
        cv = evaluator.get_splitter(D)

        self.assertEqual(cv.get_n_splits(), 1)
        train_samples, test_samples = next(cv.split(D.data['Y_train'],
                                                    D.data['Y_train']))
        self.assertEqual(len(train_samples), 6)
        self.assertEqual(len(test_samples), 3)


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

        tmp_dir_name = self.id()
        self.ev_path = os.path.join(this_directory, '.tmp_evaluations', tmp_dir_name)
        if os.path.exists(self.ev_path):
            shutil.rmtree(self.ev_path)
        os.makedirs(self.ev_path, exist_ok=False)
        self.backend = unittest.mock.Mock()
        self.backend.temporary_directory = tempfile.gettempdir()
        self.backend.get_model_dir.return_value = self.ev_path
        self.backend.get_cv_model_dir.return_value = self.ev_path
        dummy_model_files = [os.path.join(self.ev_path, str(n)) for n in range(100)]
        dummy_pred_files = [os.path.join(self.ev_path, str(n)) for n in range(100, 200)]
        dummy_cv_model_files = [os.path.join(self.ev_path, str(n)) for n in range(200, 300)]
        self.backend.get_model_path.side_effect = dummy_model_files
        self.backend.get_cv_model_path.side_effect = dummy_cv_model_files
        self.backend.get_prediction_output_path.side_effect = dummy_pred_files
        self.backend.load_datamanager.return_value = self.data
        self.backend.output_directory = 'duapdbaetpdbe'
        self.dataset_name = json.dumps({'task_id': 'test'})
        self.port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

    def tearDown(self):
        if os.path.exists(self.ev_path):
            os.rmdir(self.ev_path)

    def test_eval_holdout(self):
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        info = read_queue(self.queue)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info[0]['loss'], 0.030303030303030276, places=3)
        self.assertEqual(info[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', info[0]['additional_run_info'])

    def test_eval_holdout_all_loss_functions(self):
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=SCORER_LIST,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = read_queue(self.queue)
        self.assertEqual(len(rval), 1)

        fixture = {
            'accuracy': 0.030303030303030276,
            'balanced_accuracy': 0.033333333333333326,
            'f1_macro': 0.032036613272311221,
            'f1_micro': 0.030303030303030276,
            'f1_weighted': 0.030441716940572849,
            'log_loss': 0.06731761928478425,
            'precision_macro': 0.02777777777777779,
            'precision_micro': 0.030303030303030276,
            'precision_weighted': 0.027777777777777901,
            'recall_macro': 0.033333333333333326,
            'recall_micro': 0.030303030303030276,
            'recall_weighted': 0.030303030303030276,
            'num_run': 1,
            'validation_loss': 0.0,
            'test_loss': 0.04,
            'train_loss': 0.0,
        }

        additional_run_info = rval[0]['additional_run_info']
        for key, value in fixture.items():
            self.assertAlmostEqual(additional_run_info[key], fixture[key],
                                   msg=key)
        self.assertIn('duration', additional_run_info)
        self.assertEqual(len(additional_run_info), len(fixture) + 1,
                         msg=sorted(additional_run_info.items()))

        self.assertAlmostEqual(rval[0]['loss'], 0.030303030303030276, places=3)
        self.assertEqual(rval[0]['status'], StatusType.SUCCESS)

    def test_eval_holdout_iterative_fit_no_timeout(self):
        eval_iterative_holdout(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = read_queue(self.queue)
        self.assertEqual(len(rval), 9)
        self.assertAlmostEqual(rval[-1]['loss'], 0.030303030303030276)
        self.assertEqual(rval[0]['status'], StatusType.DONOTADVANCE)
        self.assertEqual(rval[-1]['status'], StatusType.SUCCESS)

    def test_eval_holdout_budget_iterations(self):
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
            budget=1,
            budget_type='iterations'
        )
        info = read_queue(self.queue)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info[0]['loss'], 0.06060606060606055, places=3)
        self.assertEqual(info[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', info[0]['additional_run_info'])

    def test_eval_holdout_budget_iterations_converged(self):
        configuration = get_configuration_space(
            exclude_estimators=['random_forest', 'liblinear_svc'],
            info={'task': MULTICLASS_CLASSIFICATION, 'is_sparse': False},
        ).get_default_configuration()
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude={'classifier': ['random_forest', 'liblinear_svc']},
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
            budget=80,
            budget_type='iterations'
        )
        info = read_queue(self.queue)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info[0]['loss'], 0.18181818181818177, places=3)
        self.assertEqual(info[0]['status'], StatusType.DONOTADVANCE)
        self.assertNotIn('bac_metric', info[0]['additional_run_info'])

    def test_eval_holdout_budget_subsample(self):
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
            budget=30,
            budget_type='subsample'
        )
        info = read_queue(self.queue)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info[0]['loss'], 0.0)
        self.assertEqual(info[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', info[0]['additional_run_info'])

    def test_eval_holdout_budget_mixed_iterations(self):
        print(self.configuration)
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
            budget=1,
            budget_type='mixed'
        )
        info = read_queue(self.queue)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info[0]['loss'], 0.06060606060606055)
        self.assertEqual(info[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', info[0]['additional_run_info'])

    def test_eval_holdout_budget_mixed_subsample(self):
        configuration = get_configuration_space(
            exclude_estimators=['random_forest'],
            info={'task': MULTICLASS_CLASSIFICATION, 'is_sparse': False},
        ).get_default_configuration()
        self.assertEqual(configuration['classifier:__choice__'], 'liblinear_svc')
        eval_holdout(
            queue=self.queue,
            port=self.port,
            config=configuration,
            backend=self.backend,
            resampling_strategy='holdout',
            resampling_strategy_args=None,
            seed=1,
            num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude={'classifier': ['random_forest']},
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
            budget=40,
            budget_type='mixed'
        )
        info = read_queue(self.queue)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info[0]['loss'], 0.06060606060606055)
        self.assertEqual(info[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', info[0]['additional_run_info'])

    def test_eval_cv(self):
        eval_cv(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            seed=1,
            num_run=1,
            resampling_strategy='cv',
            resampling_strategy_args={'folds': 3},
            scoring_functions=None,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = read_queue(self.queue)
        self.assertEqual(len(rval), 1)
        self.assertAlmostEqual(rval[0]['loss'], 0.04999999999999997)
        self.assertEqual(rval[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', rval[0]['additional_run_info'])

    def test_eval_cv_all_loss_functions(self):
        eval_cv(
            queue=self.queue,
            port=self.port,
            config=self.configuration,
            backend=self.backend,
            seed=1,
            num_run=1,
            resampling_strategy='cv',
            resampling_strategy_args={'folds': 3},
            scoring_functions=SCORER_LIST,
            output_y_hat_optimization=True,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            metric=accuracy,
        )
        rval = read_queue(self.queue)
        self.assertEqual(len(rval), 1)

        fixture = {
            'accuracy': 0.04999999999999997,
            'balanced_accuracy': 0.05130303030303027,
            'f1_macro': 0.052793650793650775,
            'f1_micro': 0.04999999999999997,
            'f1_weighted': 0.050090909090909096,
            'log_loss': 0.12033827614970506,
            'precision_macro': 0.04963636363636359,
            'precision_micro': 0.04999999999999997,
            'precision_weighted': 0.045757575757575664,
            'recall_macro': 0.05130303030303027,
            'recall_micro': 0.04999999999999997,
            'recall_weighted': 0.04999999999999997,
            'num_run': 1,
            'validation_loss': 0.04,
            'test_loss': 0.04,
            'train_loss': 0.0,
        }

        additional_run_info = rval[0]['additional_run_info']
        for key, value in fixture.items():
            self.assertAlmostEqual(additional_run_info[key], fixture[key], msg=key)
        self.assertIn('duration', additional_run_info)
        self.assertEqual(len(additional_run_info), len(fixture) + 1,
                         msg=sorted(additional_run_info.items()))

        self.assertAlmostEqual(rval[0]['loss'], 0.04999999999999997)
        self.assertEqual(rval[0]['status'], StatusType.SUCCESS)

    # def test_eval_cv_on_subset(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_cv(queue=self.queue, config=self.configuration, data=self.data,
    #             backend=backend_api, seed=1, num_run=1, folds=5, subsample=45,
    #             with_predictions=True, scoring_functions=None,
    #             output_y_hat_optimization=True, include=None, exclude=None,
    #             disable_file_output=False)
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.063004032258064502)
    #     self.assertEqual(info[2], 1)

    def test_eval_partial_cv(self):
        results = [0.050000000000000044,
                   0.0,
                   0.09999999999999998,
                   0.09999999999999998,
                   0.050000000000000044]
        for fold in range(5):
            instance = json.dumps({'task_id': 'data', 'fold': fold})
            eval_partial_cv(
                port=self.port,
                queue=self.queue,
                config=self.configuration,
                backend=self.backend,
                seed=1,
                num_run=1,
                instance=instance,
                resampling_strategy='partial-cv',
                resampling_strategy_args={'folds': 5},
                scoring_functions=None,
                output_y_hat_optimization=True,
                include=None,
                exclude=None,
                disable_file_output=False,
                metric=accuracy,
            )
            rval = read_queue(self.queue)
            self.assertEqual(len(rval), 1)
            self.assertAlmostEqual(rval[0]['loss'], results[fold])
            self.assertEqual(rval[0]['status'], StatusType.SUCCESS)
